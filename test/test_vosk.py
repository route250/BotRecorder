import sys,os
import time
import json
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

from vosk import KaldiRecognizer, Model, SetLogLevel

sys.path.append(os.getcwd())
from BotVoice.ace_recorder import AecRecorder, AecRes, evaluate_concentration
from BotVoice.rec_util import AudioF32, save_wave, load_wave, audio_info, sin_signal, f32_to_i16
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

IGNORE_WORDS=['小','えー','ん']
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
        self.recognizer:KaldiRecognizer
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
        SetLogLevel(-1)  # VOSK起動時のログ表示を抑制
        # 音声認識器を構築
        mdl:Model = Model(model_name="vosk-model-ja-0.22")
        self.recognizer:KaldiRecognizer = KaldiRecognizer(mdl, self.sample_rate)
    
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

    def th_transcrib(self):
        try:
            self.recorder.start()
            time.sleep(0.5)
            self.recorder.play_marker()

            # 指定した期間録音
            print("録音と再生を開始します...")
            start_time:float = time.time()
            sample_count:int = 0
            last_sample:int = 0
            blank_samples:int = int(self.sample_rate*1.2)
            # log用
            while self.run and self.recorder.is_active():
                now:float = time.time()
                res:AecRes = self.recorder.get_aec_audio()
                if len(res.audio)<=0:
                    time.sleep(0.2)
                    continue

                if (now-start_time)>5.0:
                    self.recorder.save_aec_coeff(self.aec_coeff_path)

                seg_f32 = res.audio * res.mask
                sample_count+=len(seg_f32)
                
                i16 = f32_to_i16(seg_f32)
                if self.recognizer.AcceptWaveform(i16.tobytes()):
                    vosk_res:dict = json.loads( self.recognizer.FinalResult() )
                    text = vosk_res.get("text","").strip()
                    if text!='' and not text in IGNORE_WORDS:
                        print(f"[Transcrib] Final {text}")
                        self.transcrib_buffer += f" {text}"
                        last_sample = sample_count
                else:
                    vosk_res = json.loads( self.recognizer.PartialResult() )
                    text = vosk_res.get("partial","").strip()
                    if text!='' and not text in IGNORE_WORDS:
                        print(f"[Transcrib] Partial {text}")
                        if self.transcrib_buffer=='':
                            self.transcrib_buffer=' '

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
    # main_vad()
    main()
    #main_coeff_plot()
