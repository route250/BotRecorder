import sys,os
import time
import json
import wave
from io import BytesIO
from typing import Mapping
import numpy as np
import pyaudio
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment
#help(WhisperModel)

# 録音の設定
FORMAT = pyaudio.paInt16  # 16ビットの音声フォーマット
CHANNELS = 1  # モノラル
RATE = 16000  # サンプリングレート
CHUNK = 3200  # データのチャンクサイズ
RECORD_SECONDS = 30  # 録音時間（秒）

class StreamRecog:

    def __init__(self):
        # PyAudioのインスタンスを作成
        self.audio:pyaudio.PyAudio = pyaudio.PyAudio()
        # 録音したデータを格納するリスト
        self._audio_buffer:list[bytes] = []

    # コールバック関数
    def _audio_record_callback(self,in_data:bytes|None, frame_count:int, time_info:Mapping[str,float], status:int) ->tuple[bytes,int]:
        if in_data:
            self._audio_buffer.append(in_data)
            return (in_data, pyaudio.paContinue)
        else:
            return (b'',pyaudio.paAbort) # in_dataがNoneのとき、どうすればいいの？

    # 録音を開始
    def audio_start(self):
        # 録音用のストリームを開く
        self.stream = self.audio.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK,
                            stream_callback=self._audio_record_callback)
        # streamを開始
        self.stream.start_stream()

    def stop(self):
        # 録音ストリームを停止・終了
        self.stream.stop_stream()
        self.stream.close()
        # PyAudioの終了
        self.audio.terminate()

    def get_buffer(self) ->list[bytes]:
        next_buffer = []
        bf = self._audio_buffer
        self._audio_buffer=next_buffer
        return bf


model_size = "large-v3"
model_size = "tiny"
model_size = "base"
model_size = "small"
# Run on GPU with FP16
#model = WhisperModel(model_size, device="gpu", compute_type="float16")
# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")
model = WhisperModel(model_size, device="auto", cpu_threads=8, compute_type="int8")

recog:StreamRecog = StreamRecog()

l = int( RATE * CHANNELS * RECORD_SECONDS / CHUNK )

print("録音開始")
recog.audio_start()

audio_data_list:list[bytes] = []
audio_buffer:np.ndarray = np.array([], dtype=np.int16)

time.sleep(0.5)

while len(audio_data_list)<l:
    # 録音時間待機
    delta_bytes = recog.get_buffer()
    audio_data_list.extend(delta_bytes)

    audio_bytes:bytes = b''.join(delta_bytes)

    delta_i16:np.ndarray = np.frombuffer( audio_bytes, dtype=np.int16 )
    audio_i16:np.ndarray = np.concatenate((audio_buffer,delta_i16))

    audio_f32:np.ndarray = audio_i16.astype(np.float32)/32768.0
    audio_i8:np.ndarray = (audio_f32*127).astype(np.int8)
    audio_sec:float = round( len(audio_i8)/RATE, 2 )

    segments, info = model.transcribe(audio_i8, language='ja', beam_size=5)
    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    seg_list:list[Segment] = [ seg for seg in segments ]

    text_list:list[str] = []
    split_sec:float = max(seg_list[0].start-0.6,0.0) if len(seg_list)>0 else 0.0
    for idx,seg in enumerate(seg_list):

        next:float = seg_list[idx+1].start if idx+1<len(seg_list) else audio_sec
        diff:float = next - seg.end
        if diff >0.6:
            # 確定
            print("[%.2fs -> %.2fs] %s" % (seg.start, seg.end, seg.text))
            text_list.append(seg.text)
            split_sec = seg.end
        else:
            print("[%.2fs -> %.2fs] ### %s" % (seg.start, seg.end, seg.text))
            # 未確定
            break
    # if split_sec>0:
    #     split_pos:int = int(RATE*split_sec)
    #     audio_buffer = audio_i16[split_pos:]

recog.stop()
print("録音完了")

# 録音した音声データをwaveファイルに保存
wave_buffer:BytesIO = BytesIO()
with wave.open(wave_buffer, 'wb') as wave_write_stream:
    wave_write_stream.setnchannels(CHANNELS)
    wave_write_stream.setsampwidth(pyaudio.get_sample_size(FORMAT))
    wave_write_stream.setframerate(RATE)
    wave_write_stream.writeframes(b''.join(audio_data_list))
    wave_write_stream.close()

# 録音した音声ファイルを開く
wave_buffer.seek(0)
with wave.open(wave_buffer, 'rb') as wave_read_stream:
    audio = pyaudio.PyAudio()
    # 再生用のコールバック関数
    def playback_callback(in_data, frame_count, time_info, status):
        data = wave_read_stream.readframes(frame_count)
        return (data, pyaudio.paContinue)
    # 再生用のストリームを開く
    stream = audio.open(format=audio.get_format_from_width(wave_write_stream.getsampwidth()),
                        channels=wave_write_stream.getnchannels(),
                        rate=wave_write_stream.getframerate(),
                        output=True,
                        stream_callback=playback_callback)
    print("再生中...")
    # 再生を開始
    stream.start_stream()
    # 再生中のストリームがアクティブな間待機
    while stream.is_active():
        pass
    print("再生完了")
    # 再生ストリームを停止・終了
    stream.stop_stream()
    stream.close()
    # PyAudioの終了
    audio.terminate()
