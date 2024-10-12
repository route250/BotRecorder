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
from BotVoice.ace_recorder import AecRecorder, AecRes, evaluate_convergence
from BotVoice.rec_util import AudioF32, save_wave, load_wave, audio_info, sin_signal, f32_to_i16
from BotVoice.rec_util import AudioI8, AudioI16, AudioF32, EmptyF32, np_append, save_wave, load_wave, signal_ave, sin_signal
from BotVoice.segments import TranscribRes, Segment, Word
from BotVoice.bot_audio import BotAudio,RATE,CHUNK_SEC,CHUNK_LEN
from BotVoice.voice_base import VoiceBase
from BotVoice.text_to_voice import TtsEngine
from BotVoice.vosk_util import transcrib_strip, get_text

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

LLM_PROMPT:str = """音声会話型のAIのセリフを生成してください。
人間の会話ターンの時は、聞き役として、ときどきボケてください。
あなたの会話ターンなら、どんどん話題を進めましょう。
何か話したいことある？と聞く代わりに、以下のボケをかましてください。
何か面白いことを聞かずに、以下のボケをはさんでください。
人間の考えは聞くよりボケましょう。どう思う？とか聞かなくて良い。
話題がなかったら、以下のボケをやりましょう。
# ボケるとは
相手の話した単語をランダムに選んで、関係のない単語や事柄を組み合わて意外性のあるシチュエーションで会話を誘導してください。
科学技術ネタやITネタじゃなくて、日常の話題を優先してください。
自分のネタに自分で突っ込まないことが何より重要です。"""

def is_splitter(text:str) ->int:
    for w in ('、','!','?','！','？','。'):
        i = text.find(w)
        if i>=0:
            return i
    return -1

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
        self.transcrib_partial:str|None = None
        self.transcrib_id:int = 0

        self.llm_thread:Thread|None = None
        self.llm_run:int = 0

        self.tts:TtsEngine = TtsEngine( speaker=8 )

    def start(self):
        if os.path.exists(self.aec_coeff_path):
            try:
                print(f"load aec_coeff from {self.aec_coeff_path}")
                # self.recorder.load_aec_coeff(self.aec_coeff_path)
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

    def add_talk(self,mesg:str):
        print(f"[AI]{mesg.strip()}")
        trim_msg = mesg.replace("、"," ")
        trim_msg = mesg.replace("。","、")
        trim_msg = mesg.replace("・","")
        aaa,model = self.tts._text_to_audio_by_voicevox(mesg,sampling_rate=self.sample_rate)
        if aaa is not None:
            self.recorder.play(mesg,aaa, sr=self.sample_rate)

    def th_transcrib(self):
        try:
            logdir = f"tmp/logdir"
            os.makedirs(logdir,exist_ok=True)
            self.recorder.start()
            time.sleep(0.5)
            self.recorder.play_marker()

            # 指定した期間録音
            start_time:float = time.time()
            sample_count:int = 0
            last_sample:int = 0
            blank_samples:int = int(self.sample_rate*1.2)
            # log用
            logaudio:AecRes = AecRes.empty(self.sample_rate)
            logcnt:int = 0
            coeff_save_time:float = 5
            #
            while self.run and self.recorder.is_active():
                now:float = time.time()
                try:
                    res:AecRes = self.recorder.get_aec_audio()
                    if len(res.audio)<=0:
                        time.sleep(0.2)
                        continue

                    logaudio += res
                    if logaudio.duration()>30:
                        logcnt+=1
                        fname=f"{logdir}/logaudio{logcnt:03d}.npz"
                        print(f"[LOG] save {fname}" )
                        logaudio.save( fname )
                        logaudio.clear()

                    if (now-start_time)>coeff_save_time:
                        coeff_save_time += 30
                        self.recorder.save_aec_coeff(self.aec_coeff_path)

                    seg_f32 = res.audio * res.mask * 1.5
                    sample_count+=len(seg_f32)
                    
                    i16 = f32_to_i16(seg_f32)
                    if self.recognizer.AcceptWaveform(i16.tobytes()):
                        vosk_res:dict = json.loads( self.recognizer.FinalResult() )
                        text = get_text( vosk_res )
                        if text is not None and len(text)>0:
                            print(f"[Transcrib] Final {text}")
                            self.transcrib_buffer += f" {text}"
                            self.transcrib_partial = None
                            last_sample = sample_count
                        else:
                            if self.transcrib_partial is not None:
                                print(f"[Transcrib] reset")
                                self.transcrib_partial = None
                                self.recorder.pause(False)
                    else:
                        vosk_res = json.loads( self.recognizer.PartialResult() )
                        text = get_text( vosk_res )
                        if text is not None and len(text)>0:
                            self.recorder.pause(True)
                            if text != self.transcrib_partial:
                                print(f"[Transcrib] Partial {text}")
                                self.transcrib_partial = text
                        else:
                            if self.transcrib_partial is not None:
                                print(f"[Transcrib] reset")
                                self.transcrib_partial = None
                                self.recorder.pause(False)

                    if last_sample>0 and (sample_count-last_sample)>blank_samples:
                        self.recorder.cancel()
                        self.recorder.pause(False)
                        self.transcrib_result.append(self.transcrib_buffer)
                        self.transcrib_buffer = ''
                        self.transcrib_id += 1
                        last_sample = 0
                        print(f"[Transcrib] Enter")
                        for t in self.transcrib_result:
                            print(f"   {t}")
                finally:
                    xt:float = time.time()-now
                    xt = max(.5-xt,.1)
                    time.sleep(xt)
        except:
            print_exc()
        finally:
            self.recorder.stop()
            self.run = False

    def _is_llm_cancel(self) ->bool:
        return self.transcrib_id != self.llm_run or self.transcrib_buffer!=''

    def _is_llm_abort(self) ->bool:
        return not self.run or self._is_llm_cancel()

    def th_get_response_from_openai(self, user_input):
        """
        OpenAIのAPIを使用してテキスト応答を取得する関数です。
        """
        self.global_messages = []

        # OpenAI APIの設定値
        openai_timeout = 5.0  # APIリクエストのタイムアウト時間
        openai_max_retries = 2  # リトライの最大回数
        openai_llm_model = 'gpt-4o-mini'  # 使用する言語モデル
        openai_llm_model = 'gpt-4o'  # 使用する言語モデル
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

        sentense:str = ""
        try:
            # AIの応答を取得します
            for part in stream:
                if self._is_llm_abort():
                    break
                delta_response:str|None = part.choices[0].delta.content
                if delta_response:
                    sentense += delta_response
                    if is_splitter( delta_response )>=0:
                        self.add_talk(sentense)
                        sentense = ''
        finally:
            if not self._is_llm_abort():
                if len(sentense)>0:
                    self.add_talk(sentense)
            else:
                print(f"[LLM]!!!abort!!!")
                self.recorder.cancel()

    def _th_main(self):
        try:
            while self.run:
                if self.transcrib_id == self.llm_run:
                    time.sleep(0.5)
                    continue
                assistant_output = self.recorder.get_play_text()
                print(f"[PLAYTEST]{assistant_output}")
                self.global_messages.append( {"role": "assistant", "content": assistant_output} )
                user_input = '\n'.join(self.transcrib_result)
                self.transcrib_result = []
                self.llm_run = self.transcrib_id
                self.th_get_response_from_openai(user_input)
                while self.recorder.is_playing():
                    time.sleep(0.2)
                    # if self._is_llm_abort():
                    #     print(f"[LLM]!!!cancel!!!")
                    #     self.recorder.cancel()
                    #     break
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
    abs_coeff,idx = evaluate_convergence(aec_coeff)
    print(f" {abs_coeff}")
    plt.figure()
    plt.plot(aec_coeff, label='aec_coeff', alpha=0.5)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # main_vad()
    main()
    #main_coeff_plot()
