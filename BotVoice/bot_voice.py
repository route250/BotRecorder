import sys,os
import time
from threading import Lock

import pyaudio
from pyaudio import Stream
import wave
import numpy as np

import matplotlib.pyplot as plt

from .rec_util import AudioF32, EmptyF32, np_append, save_wave, load_wave, signal_ave, sin_signal

# 録音/再生の設定
PA_FORMAT = pyaudio.paFloat32
CHANNELS:int = 1
RATE:int = 16000  # 16kHz
CHUNK_SEC:float = 0.2 # 0.2sec
CHUNK_LEN:int = int(RATE*CHUNK_SEC)

import numpy as np

class BotVoice:

    def __init__(self):
        # Lock
        self._lock:Lock = Lock()
        # PyAudioオブジェクトの作成
        self._paudio:pyaudio.PyAudio|None = pyaudio.PyAudio()
        self._stream:Stream|None = None
        # 録音設定
        self._rec_boost:float = 3.0
        # 再生用
        self._play_byffer_list:list[AudioF32] = []
        self._play_buffer:AudioF32|None = None
        self._play_pos:int = 0
        # 録音データの保存用
        self._rec_list:list[AudioF32] = []

    def is_playing(self) ->int:
        with self._lock:
            n:int = len(self._play_byffer_list)
            if self._play_buffer is not None:
                n+=1
            return n

    def is_active(self) ->bool:
        with self._lock:
            if self._stream and self._paudio:
                return self._stream.is_active()
            else:
                print("not open?")
                return False
    
    def is_stopped(self):
        with self._lock:
            if self._stream and self._paudio:
                return self._stream.is_stopped()
            else:
                return True

    def add_play(self, data:AudioF32):
        with self._lock:
            if self._play_buffer is None:
                self._play_buffer = data.copy()
                self._play_pos = 0
            else:
                self._play_byffer_list.append(data.copy())

    def start(self):
        self.stop()
        # ストリームを開く
        pa = pyaudio.PyAudio()
        st = pa.open(format=PA_FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            output=True,
                            frames_per_buffer=CHUNK_LEN,
                            stream_callback=self._audio_callback)
        with self._lock:
            self._paudio = pa
            # ストリームを開始
            self._stream = st
            st.start_stream()

    def stop(self):
        # ストリームを停止・終了
        with self._lock:
            st = self._stream
            self._stream = None
            pa = self._paudio
            self._paudio = None
        try:
            if st is not None:
                st.stop_stream()
        except:
            pass
        try:
            if st is not None:
                st.close()
        except:
            pass
        try:
            if pa is not None:
                pa.terminate()
        except:
            pass

    # コールバック関数の定義
    def _audio_callback(self, in_bytes:bytes|None, frame_count, time_info, status) ->tuple[bytes,int]:
        if frame_count != CHUNK_LEN:
            print(f"ERROR:pyaudio callback invalid frame_count != {CHUNK_LEN}")
            return b'',pyaudio.paAbort
        if in_bytes is None:
            print(f"ERROR:pyaudio callback invalid in_data is None")
            return b'',pyaudio.paAbort
        if len(in_bytes)!=CHUNK_LEN*4:
            print(f"ERROR:pyaudio callback invalid in_data len:{len(in_bytes)}")
            return b'',pyaudio.paAbort
        if status:
            print(f"status:{status}")
        # 録音データ
        in_f32:AudioF32 = np.frombuffer( in_bytes, dtype=np.float32 )

        with self._lock:
            # 録音データ
            self._rec_list.append( in_f32 )
            # 再生用データ
            play_f32 = np.zeros(CHUNK_LEN, dtype=np.float32)
            p:int = 0
            while p<CHUNK_LEN and self._play_buffer is not None:
                l:int = min(CHUNK_LEN-p, len(self._play_buffer)-self._play_pos)
                play_f32[p:p+l] = self._play_buffer[self._play_pos:self._play_pos+l]
                p+=l
                self._play_pos +=l
                if len(self._play_buffer)<=self._play_pos:
                    self._play_buffer = self._play_byffer_list.pop(0) if len(self._play_byffer_list)>0 else None
                    self._play_pos = 0

        return (play_f32.tobytes(),pyaudio.paContinue)

    def get_audio(self) ->AudioF32:
        delayc:int = 2
        with self._lock:
            es:int = delayc*3
            if len(self._rec_list)<es:
                return EmptyF32
            # 録音データ
            rec_buf = self._rec_list
            self._rec_list = []
        # 録音データ
        raw_f32:AudioF32 = np.concatenate(rec_buf)
        if 0.0<self._rec_boost<=10.0:
            raw_f32 = raw_f32 * self._rec_boost

        return raw_f32

