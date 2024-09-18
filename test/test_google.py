import sys,os
import time
from datetime import datetime

from threading import Thread, Lock
import numpy as np

import queue
import re
import time
from typing import Iterable, Iterator, Generator,Any
import json

import pyaudio

from google.cloud import speech
from google.cloud.speech_v1.types import StreamingRecognizeRequest,StreamingRecognizeResponse

from httpx import Timeout
from openai import OpenAI, OpenAIError
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionSystemMessageParam, ChatCompletionAssistantMessageParam, ChatCompletionUserMessageParam, ChatCompletionToolMessageParam, ChatCompletionToolParam
from openai.types.completion_usage import CompletionUsage
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage

sys.path.append(os.getcwd())

from BotVoice.utils import setup_openai_api, update_usage, usage_to_price, usage_to_dict, usage_to_str
from BotVoice.rec_util import AudioI8, AudioI16, AudioF32, EmptyF32, f32_to_i16, np_append, save_wave, load_wave, signal_ave, sin_signal
from BotVoice.rec_util import from_f32
from BotVoice.segments import is_accept
from BotVoice.segments import TranscribRes, Segment, Word
from BotVoice.bot_audio import BotAudio,RATE,CHUNK_SEC,CHUNK_LEN
from BotVoice.voice_base import VoiceBase


# Audio recording parameters
STREAMING_LIMIT = 240000  # 4 minutes
SAMPLE_RATE = 16000
REQ_CHUNK_LEN = int(SAMPLE_RATE / 10)  # 100ms

ERACE = "\033[K"
RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[0;33m"


def get_current_time() -> int:
    """Return Current Time in MS.

    Returns:
        int: Current Time in MS.
    """

    return int(round(time.time() * 1000))


class ResumableMicrophoneStream:
    """Opens a recording stream as a generator yielding the audio chunks."""

    def __init__(
        self,
        rate: int,
        chunk_size: int,
    ) -> None:
        self._lock:Lock = Lock()
        self._rate:int = rate
        self.chunk_size:int = chunk_size
        self._num_channels:int = 1
        self._buff:queue.Queue[AudioF32] = queue.Queue()
        self.closed:bool = True
        self.start_time = get_current_time()
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paFloat32,
            channels=self._num_channels,
            rate=self._rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            # Run the audio stream asynchronously to fill the buffer object.
            # This is necessary so that the input device's buffer doesn't
            # overflow while the calling thread makes network requests, etc.
            stream_callback=self._audio_callback,
        )

    def __enter__(self):
        """Opens the stream.

        Args:
        self: The class instance.

        returns: None
        """
        self.closed = False
        return self

    def __exit__(
        self,
        type: object,
        value: object,
        traceback: object,
    ):
        """Closes the stream and releases resources.

        Args:
        self: The class instance.
        type: The exception type.
        value: The exception value.
        traceback: The exception traceback.

        returns: None
        """
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()

    # コールバック関数の定義
    def _audio_callback(self, in_bytes:bytes|None, frame_count, time_info, status) ->tuple[bytes|None,int]:
        if frame_count != self.chunk_size:
            print(f"ERROR:pyaudio callback invalid frame_count != {self.chunk_size}")
            return b'',pyaudio.paAbort
        if in_bytes is None:
            print(f"ERROR:pyaudio callback invalid in_data is None")
            return b'',pyaudio.paAbort
        if len(in_bytes)!=self.chunk_size*4:
            print(f"ERROR:pyaudio callback invalid in_data len:{len(in_bytes)}")
            return b'',pyaudio.paAbort
        if status:
            print(f"status:{status}")
        # 録音データ
        in_f32:AudioF32 = np.frombuffer( in_bytes, dtype=np.float32 )

        self._buff.put(in_f32)
        return None, pyaudio.paContinue

    def generator(self) ->Generator[AudioF32,None,None]:
        """Stream Audio from microphone to API and to local buffer

        Args:
            self: The class instance.

        returns:
            The data from the audio stream.
        """
        while not self.closed:
            data:list[AudioF32] = []

            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk:AudioF32 = self._buff.get()

            if chunk is None:
                return
            data.append(chunk)
            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)

                    if chunk is None:
                        return
                    data.append(chunk)

                except queue.Empty:
                    break

            yield np.concatenate( data, axis=0 )

class AudioSubIterator(Iterable[StreamingRecognizeRequest]):
    """オーディオを区切った区間のイテレータ"""
    def __init__(self,iter):
        self.iter = iter
        self.stop:bool = False
    def __iter__(self):
        return self
    def __next__(self):
        try:
            if self.iter and not self.stop:
                ret = self.iter.__next__()
                if ret is not None:
                    return ret
        except:
            pass
        self.iter=None
        raise StopIteration()

class AudioMainIterator(Iterable[AudioSubIterator]):
    """オーディオ入力全体のイテレータ"""
    def __init__(self, mic_manager:ResumableMicrophoneStream ):
        self.stream:ResumableMicrophoneStream = mic_manager
        self._iter = self._request_generator()
        self._closed:bool = False

    def _request_generator(self) ->Generator[StreamingRecognizeRequest,None,None]:
        with self.stream as stream:
            while not stream.closed:
                audio_generator = stream.generator()
                if audio_generator:
                    for content in audio_generator:
                        if content is not None:
                            audio_bytes:bytes = f32_to_i16(content).tobytes()
                            yield StreamingRecognizeRequest(audio_content=audio_bytes)
        self._closed = True
        print(f"[Mic]Closed")

    def __iter__(self):
        return self

    def __next__(self):
        if self._closed:
            raise StopIteration()
        #print(':',end="")
        return AudioSubIterator(self._iter)

class RecgIterator(Iterable[StreamingRecognizeResponse]):
    """音声認識結果のイテレータ"""
    def __init__(self, mic_manager:ResumableMicrophoneStream):
        self.stream:ResumableMicrophoneStream = mic_manager
        self.client =speech.SpeechClient()
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=SAMPLE_RATE,
            language_code="ja-JP",
            max_alternatives=1,
            model='command_and_search',use_enhanced=True,
            enable_automatic_punctuation=True,  # 句読点の自動挿入を有効にする
        )
        self.streaming_config = speech.StreamingRecognitionConfig(
            config=config, interim_results=True,single_utterance=True
        )

    def __iter__(self):
        return self.ResStream()
    
    def ResStream(self) ->Generator[StreamingRecognizeResponse,None,None]:
        full_iter:AudioMainIterator = AudioMainIterator(self.stream)
        for sub_iter in full_iter:
            res_iter:Iterator[StreamingRecognizeResponse] = self.client.streaming_recognize(self.streaming_config, sub_iter )
            for response in res_iter:
                eos = response.speech_event_type == StreamingRecognizeResponse.SpeechEventType.END_OF_SINGLE_UTTERANCE
                is_final = response.results[0].is_final if len(response.results)>0 else False
                if eos:
                    sub_iter.stop=True
                # if is_final:
                #     print("<Final>",end="")
                # elif eos:
                #     print("<EOS>",end="")
                # else:
                #     print(".",end="")
                yield response
            #print(f"\r{ERACE}",end="")

TRANSCRIPT_JSON_PATH='tmp/transcript.json'
PROMPT_TEXT_PATH='tmp/prompt.txt'
MEETING_MINUTES_TEXT_PATH='tmp/meeting_minutes.md'

def get_dt() ->str:
    # 現在の日時を取得し、指定したフォーマットで文字列に変換
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    return current_time

def get_file_time(filepath:str) ->float:
    # ファイルの最終更新時刻を取得
    try:
        return os.path.getmtime(filepath)
    except:
        pass
    return 0.0

def load_json(filepath:str,default=None):
    """ファイルをロードして処理する関数"""
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data
    except Exception as ex:
        print(ex)
    return default

def save_json(filepath:str,data):
    # ファイルにデータを書き込む
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4) 

def load_text(filepath:str,default:str|None=None) ->str:
    """ファイルをロードして処理する関数"""
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            data = file.read()
            return data
    except Exception as ex:
        print(ex)
    return default

def save_text(filepath:str,data:str):
    # ファイルにデータを書き込む
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(data)

def get_transcription_time() ->float:
    return get_file_time(TRANSCRIPT_JSON_PATH)

def load_transcription() ->list[dict]:
    return load_json(TRANSCRIPT_JSON_PATH,[])

def save_transcription(data:list[dict]):
    save_json(TRANSCRIPT_JSON_PATH, data)

def get_prompt_time() ->float:
    return get_file_time(PROMPT_TEXT_PATH)

def load_prompt() ->str:
    return load_text(PROMPT_TEXT_PATH,"")

def get_meeting_minutes_time() ->float:
    return get_file_time(MEETING_MINUTES_TEXT_PATH)

def load_meeting_minutes() ->str:
    return load_text(MEETING_MINUTES_TEXT_PATH, None)

def save_meeting_minutes(data:str):
    save_text(MEETING_MINUTES_TEXT_PATH, data)

def main_transcription() -> None:
    """start bidirectional streaming from microphone input to speech API"""

    mic_stream:ResumableMicrophoneStream = ResumableMicrophoneStream(SAMPLE_RATE, REQ_CHUNK_LEN*1)

    sys.stdout.write(YELLOW)
    sys.stdout.write('\nListening, say "Quit" or "Exit" or "終了" to stop.\n\n')
    sys.stdout.write("End (ms)       Transcript Results/Status\n")
    sys.stdout.write("=====================================================\n")

    data_array:list[dict] = load_transcription()
    for x in data_array:
        sys.stdout.write(str(x.get('tm',0)) + ": " + x.get('content','') + "\n")
    recg_iter:RecgIterator = RecgIterator(mic_stream)
    for response in recg_iter:

        if not response.results:
            continue

        result = response.results[0]

        if not result.alternatives:
            continue

        transcript = result.alternatives[0].transcript
        print("\r",end="")

        result_seconds = 0
        result_micros = 0

        if result.result_end_time.seconds:
            result_seconds = result.result_end_time.seconds

        if result.result_end_time.microseconds:
            result_micros = result.result_end_time.microseconds

        result_end_time = int((result_seconds * 1000) + (result_micros / 1000))

        corrected_time = result_end_time
        # Display interim results, but with a carriage return at the end of the
        # line, so subsequent lines will overwrite them.
        if result.is_final:
            sys.stdout.write(GREEN)
            sys.stdout.write(ERACE)
            sys.stdout.write(str(corrected_time) + ": " + transcript + "\n")
            data_array.append( { 'tm': corrected_time, 'content': transcript } )
            save_transcription(data_array)

            # Exit recognition if any of the transcribed phrases could be
            # one of our keywords.
            if re.search(r"\b(exit|quit|終了)\b", transcript, re.I):
                sys.stdout.write(YELLOW)
                sys.stdout.write("Exiting...\n")
                mic_stream.closed = True
                break
        else:
            sys.stdout.write(RED)
            sys.stdout.write(ERACE)
            sys.stdout.write(str(corrected_time) + ": " + transcript + "\r")

def main_llm():

    """ファイルの更新を監視し、更新があれば処理する関数"""

    usage:CompletionUsage = CompletionUsage(completion_tokens=0,prompt_tokens=0,total_tokens=0)

    last_mtime = 0 # get_transcription_time()
    last_prompt_time = get_prompt_time()
    
    while True:
        time.sleep(1.0)  # 監視間隔を指定（秒単位）
        # ファイルの最終更新時刻を取得
        current_mtime = get_transcription_time()
        current_prompt_time = get_prompt_time()
        if current_prompt_time==last_prompt_time and (current_mtime-last_mtime)<10.0:
            continue
        last_mtime = current_mtime
        last_prompt_time = current_prompt_time
        data = load_transcription()
        if data is None:
            continue
        prompt = load_prompt()
        print("\nstart llm\n")
        now_dt:str = get_dt()
        aatxt:str = json.dumps(data,ensure_ascii=False)
        # OpenAI APIを使って議事録の要約を作成
        req_messages=[
            {"role": "system", "content": f"current time is {now_dt}\nYou are an assistant who helps summarize meeting transcripts into concise minutes.\n\n{prompt}"},
            {"role": "user", "content": f"以下が会議内容:\n\n{aatxt}"}
        ]
        try:
            client:OpenAI = OpenAI()# timeout=self.openai_timeout,max_retries=1)
            response:ChatCompletion = client.chat.completions.create(
                    messages=req_messages,
                    model='gpt-4o-mini',
                    temperature=0.7,
            )
            update_usage(usage,response.usage)
            ch:Choice = response.choices[0]
            msg:ChatCompletionMessage = ch.message
            #print(msg.content)
            content = f"{usage_to_str(usage)}"
            if msg.content:
                content += "\n\n" + msg.content
            save_meeting_minutes(content)
        except Exception as e:
            print(f"エラーが発生しました: {e}")
            return None        

def main():
    th1 = Thread( name='recog', target=main_transcription )
    th2 = Thread( name='llm', target=main_llm )
    
    threads = (th1,th2)
    for t in threads:
        t.start()
    for t in threads:
        t.join()

if __name__ == "__main__":
    setup_openai_api()
    os.environ['GOOGLE_APPLICATION_CREDENTIALS']='hidden-chiller-434114-t2-4d39fc610ee8.json'
    main()

# [END speech_transcribe_infinite_streaming]
