import sys,os
import time

import pyaudio
from pyaudio import Stream
import wave
import numpy as np
from scipy.signal import resample

# 読み込むWaveファイルの設定
input_wave_filename = 'input.wav'
output_recorded_filename = 'recorded.wav'
output_subtracted_filename = 'subtracted.wav'

# 録音/再生の設定
PA_FORMAT = pyaudio.paFloat32
CHANNELS:int = 1
RATE:int = 16000  # 16kHz
CHUNK_SEC:float = 0.2 # 0.2sec
CHUNK_LEN:int = int(RATE*CHUNK_SEC)
DURATION:float = 1.0

class EchoLessRecorder:

    def __init__(self):
        # PyAudioオブジェクトの作成
        self._paudio:pyaudio.PyAudio = pyaudio.PyAudio()
        self._stream:Stream

        # 再生用
        self._play_list:list[np.ndarray] = []
        self._play_buffer:np.ndarray|None = None
        self._play_pos:int = 0
        self._echo_buffer:list[np.ndarray] = []

        # 録音データの保存用
        self._rec_buffer:list[np.ndarray] = []

    def is_active(self):
        return self._stream and self._stream.is_active()
    
    def is_stopped(self):
        return self._stream and self._stream.is_stopped()

    def add_play(self, data:np.ndarray):
        self._play_list.append(data)

    def start(self):

        # ストリームを開く
        self._stream = self._paudio.open(format=PA_FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        output=True,
                        frames_per_buffer=CHUNK_LEN,
                        stream_callback=self._audio_callback)
        # ストリームを開始
        self._stream.start_stream()

    def stop(self):
        # ストリームを停止・終了
        self._stream.stop_stream()
        self._stream.close()
        self._paudio.terminate()

    # コールバック関数の定義
    def _audio_callback(self, in_data:bytes|None, frame_count, time_info, status) ->tuple[bytes,int]:
        # 録音データ
        if in_data:
            self._rec_buffer.append( np.frombuffer(in_data, dtype=np.float32) )

        # 再生用データ
        play_data = np.zeros(CHUNK_LEN, dtype=np.float32)
        if self._play_buffer is not None or len(self._play_list)>0:
            p:int = 0
            while p<CHUNK_LEN:
                if self._play_buffer is None:
                    if len(self._play_list)>0:
                        self._play_buffer = self._play_list.pop(0)
                        self._play_pos = 0
                    else:
                        break
                if self._play_buffer is not None:
                    l:int = min(CHUNK_LEN-p, len(self._play_buffer)-self._play_pos)
                    play_data[p:p+l] = self._play_buffer[self._play_pos:self._play_pos+l]
                    p+=1
                    self._play_pos +=1
                    if len(self._play_buffer)<=self._play_pos:
                        self._play_buffer = None
                        self._play_pos = 0
        self._echo_buffer.append( play_data )
        return (play_data.tobytes(),pyaudio.paContinue)

    def get_audio(self) ->tuple[np.ndarray,np.ndarray]:
        # 録音データをnumpy配列に変換
        ia = self._rec_buffer
        self._rec_buffer = []
        ib = self._echo_buffer
        self._echo_buffer = [ ib[-1] ] if len(ib)>0 else []
        recorded_data = np.concatenate(ia)
        echo_data = np.concatenate(ib)

        # 再生音を引き算してノイズ除去
        min_diff = None
        best_offset = 0
        best_subtracted_data = None

        # 最適なオフセットを探す
        for offset in range(-CHUNK_LEN, CHUNK_LEN, int(CHUNK_LEN / 10)):  # -CHUNKから+CHUNKまでの範囲を探す
            adjusted_play_data = np.roll(echo_data, offset)
            subtracted_data = recorded_data[:len(adjusted_play_data)] - adjusted_play_data
            volume = np.sum(np.abs(subtracted_data))

            if min_diff is None or volume < min_diff:
                min_diff = volume
                best_offset = offset
                best_subtracted_data = subtracted_data

        print(f"最適なオフセット: {best_offset} サンプル")
        return best_subtracted_data,recorded_data

# WAVファイルとして保存
def save_wave(filename, data, rate):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes( (data*32767).astype(np.int16).tobytes())

def main():
    elrec:EchoLessRecorder = EchoLessRecorder()

    # 再生音をnumpy配列に読み込む
    with wave.open(input_wave_filename, 'rb') as iw:
        wave_bytes = iw.readframes(iw.getnframes())
        original_data = np.frombuffer(wave_bytes, dtype=np.int16).astype(np.float32)/32768.0
        # 16kHzにリサンプリング（必要ならば）
        if iw.getframerate() != RATE:
            original_data = resample(original_data, int(len(original_data) * RATE / iw.getframerate()))
    elrec.add_play( original_data )

    print("録音と再生を開始します...")
    elrec.start()

    filterd_audio_list = []
    raw_audio_list = []
    # 指定した期間録音
    while elrec.is_active():
        a,b = elrec.get_audio()
        filterd_audio_list.append(a)
        raw_audio_list.append(b)
        time.sleep( DURATION )  # ミリ秒単位で指定

    # 生の録音音声を保存
    recorded_data = np.concat( filterd_audio_list )
    save_wave(output_recorded_filename, recorded_data, RATE)

    # 引き算後の音声を保存
    best_subtracted_data = np.concat(raw_audio_list )
    save_wave(output_subtracted_filename, best_subtracted_data, RATE)

    print(f"録音されたデータが {output_recorded_filename} に保存されました。")
    print(f"引き算されたデータが {output_subtracted_filename} に保存されました。")

if __name__ == "__main__":
    main()