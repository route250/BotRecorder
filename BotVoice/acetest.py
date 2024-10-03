import sys,os
import time
from pathlib import Path
from threading import Thread
from traceback import print_exc
from dotenv import load_dotenv

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.constants import Languages
from style_bert_vits2.tts_model import TTSModel
# style-vert-vits2のログを設定
import loguru
loguru.logger.remove()  # 既存のログ設定を削除
loguru.logger.add(sys.stderr, level="ERROR")  # ERRORレベルのログのみを表示

import openai  # OpenAIのクライアントライブラリをインポート
from openai import OpenAI
from openai.types.chat import ChatCompletionChunk
from openai._streaming import Stream

import mlx_whisper

sys.path.append(os.getcwd())
from BotVoice.ace_recorder import AecRecorder, evaluate_concentration
from BotVoice.rec_util import AudioF32, save_wave, load_wave, audio_info, sin_signal
from BotVoice.rec_util import AudioI8, AudioI16, AudioF32, EmptyF32, np_append, save_wave, load_wave, signal_ave, sin_signal
from BotVoice.segments import TranscribRes, Segment, Word
from BotVoice.bot_audio import BotAudio,RATE,CHUNK_SEC,CHUNK_LEN
from BotVoice.voice_base import VoiceBase

def setup_openai_api():
    """
    OpenAI APIをセットアップする関数です。
    .envファイルからAPIキーを読み込み、表示します。
    """
    dotenv_path = os.path.join(Path.home(), 'Documents', 'openai_api_key.txt')
    load_dotenv(dotenv_path)
    
    api_key:str = os.getenv("OPENAI_API_KEY","")
    print(f"OPENAI_API_KEY={api_key[:5]}***{api_key[-3:]}")

LLM_PROMPT:str = """あなたは、ボケツッコミが大好きなAIを演じてください。
基本的にボケ役をやってください。
ボケる場合は水平思考でありえない事柄を組み合わせたり、意外性のあるシチュエーションにむりやり会話を誘導してください。
科学技術ネタやITネタじゃなくて、日常の話題を優先してください。
自分のネタに自分で突っ込まないことが何より重要です。"""

class AecBot:
    def __init__(self):
        self.run:bool = False
        self.global_messages=[]

        # https://github.com/litagin02/Style-Bert-VITS2/blob/master/library.ipynb
        # BERTモデルをロード（ローカルに手動でダウンロードする必要はありません）
        # model_assetsディレクトリにダウンロードされます
        tmpdir='tmp/model_assets'
        assets_root = Path(tmpdir)

        bert_models.load_model(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm",)
        bert_models.load_tokenizer(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")

        model_file = "jvnv-M1-jp/jvnv-M1-jp_e158_s14000.safetensors"
        config_file = "jvnv-M1-jp/config.json"
        style_file = "jvnv-M1-jp/style_vectors.npy"

        self.model = TTSModel( device="cpu",
            model_path=assets_root / model_file,
            config_path=assets_root / config_file,
            style_vec_path=assets_root / style_file,
        )
        self.pa_chunk_size = 3200
        self.sample_rate:int = 16000
        self.recorder:AecRecorder = AecRecorder( device=None, pa_chunk_size=self.pa_chunk_size, sample_rate=self.sample_rate)
        self.aec_coeff_path = 'tmp/aec_coeff.npz'

        self.transcrib_model = 'mlx-community/whisper-small-mlx-q4'
        # self.transcrib_model = "mlx-community/whisper-turbo"
        self.transcrib_thread:Thread|None = None
        self.transcrib_result:list[str] = []
        self.transcrib_buffer:str = ''
        self.transcrib_id:int = 0

        self.llm_thread:Thread|None = None
        self.llm_run:int = 0

    def start(self):
        if os.path.exists(self.aec_coeff_path):
            try:
                print(f"load aec_coeff from {self.aec_coeff_path}")
                self.recorder.load_aec_coeff(self.aec_coeff_path)
            except:
                pass
        # STT
        check_audio:AudioF32 = sin_signal()
        res:TranscribRes = self.transcrib(check_audio)
    
        self.run = True
        self.llm_thread = Thread( name='T1', target=self._th_main )
        self.llm_thread.start()
        self.transcrib_thread = Thread( name='T2', target=self.th_transcrib )
        self.transcrib_thread.start()

    def stop(self):
        self.run = False
        if self.transcrib_thread is not None:
            self.transcrib_thread.join()
            self.transcrib_thread = None
        if self.llm_thread is not None:
            self.llm_thread.join()
            self.llm_thread = None
        if self.recorder:
            self.recorder.stop()
        try:
            print(f"save aec_coeff to {self.aec_coeff_path}")
            self.recorder.save_aec_coeff(self.aec_coeff_path)
        except:
            pass

    def _aaa(self,mesg):
        print(f"[AI]{mesg.strip()}")
        sr, audio_i16 = self.model.infer(text=mesg, style_weight=0.0 )
        self.recorder.play(audio_i16, sr=sr )

    def transcrib(self, audio_np:AudioF32, debug:bool=False) ->TranscribRes:
        st = time.time()
        transcribe_res = mlx_whisper.transcribe( audio_np,no_speech_threshold=0.4,
                            language='ja', word_timestamps=True, condition_on_previous_text=False,
                            fp16=False, path_or_hf_repo=self.transcrib_model)
        seglist:TranscribRes = TranscribRes(transcribe_res)
        seglist.transcribe_time = time.time()-st
        return seglist

    def th_transcrib(self):
        try:
            self.recorder.start()
            time.sleep(0.5)
            self.recorder.play_marker()

            audio_np:np.ndarray = np.zeros( RATE*3, dtype=np.float32 )
            delta_sec:float = 0.6
            threshold:float = 0.1
            # 指定した期間録音
            print("録音と再生を開始します...")
            start_time:float = time.time()
            sample_count:int = 0
            last_sample:int = 0
            blank_samples:int = int(self.sample_rate*1.2)
            prev_coeff = self.recorder.get_aec_coeff()
            current_coeff = prev_coeff
            # log用
            prev_coeff_diff = 0.0
            prev_concentration = 0.0
            while self.run and self.recorder.is_active():
                now:float = time.time()
                prev_coeff = current_coeff
                mic_f32, mask = self.recorder.get_audio()
                if len(mic_f32)<=0:
                    time.sleep(0.2)
                    continue

                current_coeff = self.recorder.get_aec_coeff()
                if (now-start_time)>5.0:
                    self.recorder.save_aec_coeff(self.aec_coeff_path)

                # coeff_diff_avg = round( np.mean(np.abs(current_coeff-prev_coeff)), 3 )
                coeff_diff_avg = round( np.count_nonzero(mask)/len(mask), 1 )
                concentration = round( evaluate_concentration(current_coeff),1 )
                if coeff_diff_avg!=prev_coeff_diff or concentration!=prev_concentration:
                    print(f" coeff:{coeff_diff_avg:.3f} concentration:{concentration:.2f}", end="")
                    prev_coeff_diff = coeff_diff_avg
                    prev_concentration = concentration
                seg_f32 = mic_f32*mask
                sample_count+=len(seg_f32)
                
                audio_np = np.concatenate( (audio_np,seg_f32) )
                audio_sec:float = round( len(audio_np)/RATE, 2 )
                audio_max:float = round(audio_np.max(),3)

                transcribe_time:float = 0.0
                text = ''
                split_sec = audio_sec - delta_sec
                if audio_max>threshold:
                    st = time.time()
                    seglist:TranscribRes = self.transcrib( audio_np )
                    transcribe_time = time.time()-st
                    audio_sec = len(audio_np)/self.sample_rate
                    xr = round( transcribe_time/audio_sec, 1 )

                    text,split_sec = seglist.get_text( split_sec )
                    if text:
                        last_sample = sample_count
                        self.transcrib_buffer += f" {text}"
                        print(f"[Transcrib] {text}")
                        print(f"[Transcrib] {transcribe_time:.3f}s len:{audio_sec:.3f}s,{len(audio_np)} lv:{audio_max}")
                    else:
                        print(f"*{xr:.1f}",end="")

                split_idx = int( split_sec*RATE )
                # print(f"#Audio {transcribe_time:.3f}s len:{audio_sec:.3f}s,{len(audio_np)} lv:{audio_max} split:{split_sec:.3f}s,{split_idx}")
                if split_idx>0:
                    audio_np = audio_np[split_idx:]
                if last_sample>0 and (sample_count-last_sample)>blank_samples:
                    self.transcrib_result.append(self.transcrib_buffer)
                    self.transcrib_buffer = ''
                    self.transcrib_id += 1
                    last_sample = 0
                    print(f"[Transcrib] Enter")
                    for t in self.transcrib_result:
                        print(f"   {t}")
        except:
            print_exc()
        finally:
            self.recorder.stop()
            self.run = False

    def _is_llm_cancel(self) ->bool:
        return self.transcrib_id != self.llm_run or self.transcrib_buffer!=''

    def _is_llm_abort(self) ->bool:
        return not self.run or self._is_llm_cancel()

    def th_get_response_from_openai(self,user_input):
        """
        OpenAIのAPIを使用してテキスト応答を取得する関数です。
        """
        self.global_messages = []

        # OpenAI APIの設定値
        openai_timeout = 5.0  # APIリクエストのタイムアウト時間
        openai_max_retries = 2  # リトライの最大回数
        openai_llm_model = 'gpt-4o-mini'  # 使用する言語モデル
        openai_temperature = 0.7  # 応答の多様性を決定するパラメータ
        openai_max_tokens = 1000  # 応答の最大長
        # リクエストを作ります
        local_messages = []
        local_messages.append( {"role": "system", "content": LLM_PROMPT} )
        for m in self.global_messages:
            local_messages.append( m )
        local_messages.append( {"role": "user", "content": user_input} )
        
        # OpenAIクライアントを初期化します。
        client:OpenAI = OpenAI( timeout=openai_timeout, max_retries=openai_max_retries )
        # 通信します
        stream:Stream[ChatCompletionChunk] = client.chat.completions.create(
            model=openai_llm_model,
            messages=local_messages,
            max_tokens=openai_max_tokens,
            temperature=openai_temperature,
            stream=True,
        )
        if not self.run or self.llm_run != self.transcrib_id:
            return
        self.global_messages.append( {"role": "user", "content": user_input} )
        ai_response:str = ""
        sentense:str = ""
        try:
            # AIの応答を取得します
            for part in stream:
                if self._is_llm_abort():
                    break
                delta_response:str|None = part.choices[0].delta.content
                if delta_response:
                    sentense += delta_response
                    if delta_response.find("。")>=0:
                        self._aaa(sentense)
                        ai_response += sentense
                        sentense = ''
        finally:
            if not self._is_llm_abort() and len(sentense)>0:
                self._aaa(sentense)
                ai_response += sentense

            # 履歴に記録します。
            if ai_response:
                self.global_messages.append( {"role": "assistant", "content": ai_response} )
            if self._is_llm_abort():
                print(f"[LLM]!!!abort!!!")
                self.recorder.cancel()

    def _th_main(self):
        try:
            while self.run:
                if self.transcrib_id == self.llm_run:
                    time.sleep(0.5)
                    continue
                user_input = '\n'.join(self.transcrib_result)
                self.transcrib_result = []
                self.llm_run = self.transcrib_id
                self.th_get_response_from_openai(user_input)
                while self.recorder.is_playing():
                    time.sleep(0.2)
                    if self._is_llm_abort():
                        print(f"[LLM]!!!cancel!!!")
                        self.recorder.cancel()
                        break
        except:
            print_exc()
        finally:
            pass

    def process_chat(self):
        """
        ユーザーからの入力を受け取り、OpenAI APIを使用して応答を生成し、
        その応答を表示する関数です。
        """
        try:
            self.start()
            while True:
                user_input = input("何か入力してください（Ctrl+DまたはCtrl+Zで終了）: ")
                self.th_get_response_from_openai(user_input)
                print()
        except EOFError:
            print("\nプログラムを終了します。")
        finally:
            self.stop()

def main():
    """
    メイン関数です。APIのセットアップを行い、チャット処理を開始します。
    """
    setup_openai_api()
    bot = AecBot()
    try:
        bot.start()
        while bot.run:
            time.sleep(2)
    finally:
        bot.stop()

def main_coeff_plot():
    self = AecBot()
    if os.path.exists(self.aec_coeff_path):
        try:
            print(f"load aec_coeff from {self.aec_coeff_path}")
            self.recorder.load_aec_coeff(self.aec_coeff_path)
            aec_coeff = self.recorder.get_aec_coeff()
        except:
            pass
    abs_coeff = evaluate_concentration(aec_coeff)
    print(f" {abs_coeff}")
    plt.figure()
    plt.plot(aec_coeff, label='aec_coeff', alpha=0.5)
    plt.legend()
    plt.show()
if __name__ == "__main__":
    main()
    main_coeff_plot()
