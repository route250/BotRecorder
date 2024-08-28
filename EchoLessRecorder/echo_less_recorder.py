import sys,os
import time
from threading import Lock

import pyaudio
from pyaudio import Stream
import wave
import numpy as np
from numpy.typing import NDArray
from scipy.signal import resample

# 型エイリアス
AudioF32 = NDArray[np.float32]
AudioF16 = NDArray[np.float16]
AudioI16 = NDArray[np.int16]
AudioI8 = NDArray[np.int8]

# 録音/再生の設定
PA_FORMAT = pyaudio.paFloat32
CHANNELS:int = 1
RATE:int = 16000  # 16kHz
CHUNK_SEC:float = 0.2 # 0.2sec
CHUNK_LEN:int = int(RATE*CHUNK_SEC)
BUFFER_LEN:int = int(RATE/CHUNK_LEN) * CHUNK_LEN
DURATION:float = 1.0

import numpy as np

def np_shiftL( a:np.ndarray, n:int=1 ):
    if 0<n and n<len(a)-1:
        a[:-n] = a[n:]

def np_append( buf:AudioF32, x:AudioF32 ):
    n:int = len(x)
    if n>=len(buf):
        buf = x[:-len(buf)]
    else:
        buf[:-n] = buf[n:]
        buf[-n:] = x

def lms_filter(desired_signal, input_signal, mu=0.01, filter_order=32):
    """
    LMS適応フィルタによるエコーキャンセル

    :param desired_signal: 望ましい信号（マイク入力信号）
    :param input_signal: 入力信号（スピーカー出力信号）
    :param mu: ステップサイズ（学習率）
    :param filter_order: フィルタの次数
    :return: フィルタ出力信号、エラー信号
    """
    n = len(desired_signal)
    w = np.zeros(filter_order)  # フィルタ係数の初期化
    y = np.zeros(n)             # 出力信号
    e = np.zeros(n)             # エラー信号

    for i in range(filter_order, n):
        x = input_signal[i:i-filter_order:-1]  # 過去の入力信号のスライス
        y[i] = np.dot(w, x)                   # フィルタ出力
        e[i] = desired_signal[i] - y[i]       # エラー計算
        w += 2 * mu * e[i] * x                # フィルタ係数の更新

    return y, e

class EchoLessRecorder:
    EmptyF32:AudioF32 = np.zeros(0,dtype=np.float32)
    ZerosF32:AudioF32 = np.zeros(CHUNK_LEN,dtype=np.float32)

    def __init__(self):
        # Lock
        self._lock:Lock = Lock()
        # PyAudioオブジェクトの作成
        self._paudio:pyaudio.PyAudio = pyaudio.PyAudio()
        self._stream:Stream

        # 再生用
        self._play_byffer_list:list[AudioF32] = []
        self._play_buffer:AudioF32|None = None
        self._play_pos:int = 0
        # エコーバックデータ保存用
        self._echo_list:list[AudioF32] = [EchoLessRecorder.ZerosF32,EchoLessRecorder.ZerosF32]
        # 録音データの保存用
        self._rec_list:list[AudioF32] = []
        self._echo_array:AudioF32 = np.zeros( BUFFER_LEN*2, dtype=np.float32 )
        self._echo_len:int = 0
        self._rec_array:AudioF32 = np.zeros( BUFFER_LEN*2, dtype=np.float32 )

        # LMSフィルタの基本設定
        self.lms_mu = 0.0001  # フィルタの学習率（調整が必要）
        self.lms_filter_order = 32  # フィルタの長さ（調整が必要）
        # 初期化
        self.w = np.zeros(self.lms_filter_order)  # フィルタの重み

    def is_playing(self) ->int:
        with self._lock:
            n:int = len(self._play_byffer_list)
            if self._play_buffer is not None:
                n+=1
            return n

    def is_active(self):
        with self._lock:
            return self._stream and self._stream.is_active()
    
    def is_stopped(self):
        with self._lock:
            return self._stream and self._stream.is_stopped()

    def add_play(self, data:AudioF32):
        with self._lock:
            if self._play_buffer is None:
                self._play_buffer = data.copy()
                self._play_pos = 0
            else:
                self._play_byffer_list.append(data.copy())

    def start(self):
        with self._lock:
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
        with self._lock:
            # ストリームを停止・終了
            self._stream.stop_stream()
            self._stream.close()
            self._paudio.terminate()

    # コールバック関数の定義
    def _audio_callback(self, in_bytes:bytes|None, frame_count, time_info, status) ->tuple[bytes,int]:
        with self._lock:
            # 録音データ
            if in_bytes:
                in_f32:AudioF32 = np.frombuffer( in_bytes, dtype=np.float32 )
                np_append( self._rec_array, in_f32 )
                self._rec_list.append( in_f32 )
            else:
                print("in_data is None")

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
            self._echo_list.append( play_f32 )
            if p>0:
                self._echo_len += p
            else:
                self._echo_len = 0
            np_append( self._echo_array, play_f32 )
        return (play_f32.tobytes(),pyaudio.paContinue)

    def get_audio(self) ->tuple[AudioF32,AudioF32]:
        delayc:int = 2
        with self._lock:
            es:int = delayc*2
            if len(self._rec_list)<es:
                return EchoLessRecorder.EmptyF32,EchoLessRecorder.EmptyF32
            # 録音データをnumpy配列に変換
            rec_buf = self._rec_list
            self._rec_list = []
            echo_buf = self._echo_list
            self._echo_list = self._echo_list[-es:]
        
        raw_audio_f32:AudioF32 = np.concatenate(rec_buf)
        echo_data:AudioF32 = np.concatenate(echo_buf)

        filtered_audio_f32 = self._apply_filter( raw_audio_f32, echo_data )
        return filtered_audio_f32,raw_audio_f32

    def _apply_filter(self, raw_audio_f32:AudioF32, echo_audio_f32:AudioF32) ->AudioF32:
        rlen:int = raw_audio_f32.shape[0]
        offset_end:int = echo_audio_f32.shape[0]-rlen
        if offset_end<0:
            raise ValueError('invalid buffer size???')

        # 移動平均のウィンドウサイズ
        window_size = 7
        # 平均化のためのカーネルを作成
        kernel = np.ones(window_size) / window_size
        # 移動平均を計算
        moving_average = np.convolve(echo_audio_f32, kernel, mode='same')

        # 再生音を引き算してノイズ除去
        min_volume:float = sys.float_info.max
        best_offset:int = 0
        best_mask = raw_audio_f32

        # 最適なオフセットを探す
        for offset in range(offset_end):  # -CHUNKから+CHUNKまでの範囲を探す
            mask_f32 = moving_average[offset:offset+rlen]
            subtracted_f32 = raw_audio_f32 - mask_f32*0.1
            volume:float = np.sum(np.abs(subtracted_f32))
            if volume < min_volume:
                min_volume = volume
                best_offset = offset
                best_mask = mask_f32

        filtered_audio_f32:AudioF32 = raw_audio_f32
        best_lv=0.1
        for i in range(1,200):
            lv:float = float(i)/100.0
            subtracted_f32 = raw_audio_f32 - best_mask*lv
            volume:float = np.sum(np.abs(subtracted_f32))
            if volume < min_volume:
                min_volume = volume
                best_lv = lv
                filtered_audio_f32 = subtracted_f32
        
        print(f" {best_offset}:{best_lv}",end="")
        # print(f"最適なオフセット: {best_offset} サンプル")
        return filtered_audio_f32

    def _apply_filter_lms(self, recorded_data: AudioF32, played_data: AudioF32) -> AudioF32:

        filtered_data:AudioF32 = np.zeros_like(recorded_data)

        for i in range(self.lms_filter_order, len(recorded_data)):
            x = played_data[i-self.lms_filter_order:i]  # 再生音データの一部
            d = recorded_data[i]  # 録音データ
            y = np.dot(self.w, x)  # フィルタ出力
            e = d - y  # エラーデータ（人間の声を含む残差）

            # フィルタの重みを更新
            self.w += 2 * self.lms_mu * e * x

            # 出力信号を記録
            filtered_data[i] = e

        return filtered_data

# WAVファイルとして保存
def save_wave(filename, data, rate):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes( (data*32767).astype(np.int16).tobytes())

def sinwave() ->AudioF32:
    # パラメータの設定
    sampling_rate = 16000  # サンプリングレート 16kHz
    frequency = 220  # 生成する音声の周波数 100Hz
    duration = 10.0  # 生成する音声の長さ（秒）
    # 時間軸の作成
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    # サイン波の生成
    audio_signal = np.sin(2 * np.pi * frequency * t)
    # データ型をfloat32に変換
    audio_signal_float32 = audio_signal.astype(np.float32)
    return audio_signal_float32

def main():

    # 読み込むWaveファイルの設定
    play_filename = 'test/testData/ttsmaker-file-2024-8-27-20-36-50.wav'
    output_raw_filename = 'tmp/raw_audio.wav'
    output_filtered_filename = 'tmp/filtered_audio.wav'

    el_recorder:EchoLessRecorder = EchoLessRecorder()

    # 再生音をnumpy配列に読み込む
    with wave.open(play_filename, 'rb') as iw:
        wave_bytes = iw.readframes(iw.getnframes())
        play_audio_f32:np.ndarray = np.frombuffer(wave_bytes, dtype=np.int16).astype(np.float32)/32768.0
        # 16kHzにリサンプリング（必要ならば）
        if iw.getframerate() != RATE:
            play_audio_f32 = resample(play_audio_f32, int(len(play_audio_f32) * RATE / iw.getframerate()))
    #play_audio_f32 = sinwave()
    el_recorder.add_play( play_audio_f32 )

    print("録音と再生を開始します...")
    el_recorder.start()

    filterd_audio_list = []
    raw_audio_list = []
    # 指定した期間録音
    while el_recorder.is_active() and el_recorder.is_playing():
        filtered_segment_f32, raw_audio_f32 = el_recorder.get_audio()
        if len(filtered_segment_f32)>0:
            filterd_audio_list.append(filtered_segment_f32)
            raw_audio_list.append(raw_audio_f32)
        time.sleep( DURATION )  # ミリ秒単位で指定

    # 生の録音音声を保存
    raw_audio_f32 = np.concatenate( raw_audio_list )
    save_wave(output_raw_filename, raw_audio_f32, RATE)

    # 引き算後の音声を保存
    filtered_audio_f32 = np.concatenate(filterd_audio_list )
    save_wave(output_filtered_filename, filtered_audio_f32, RATE)

    print(f"録音されたデータが {output_raw_filename} に保存されました。")
    print(f"引き算されたデータが {output_filtered_filename} に保存されました。")

def mlstest():

    # シミュレーション用の信号（スピーカー音とマイク音）
    fs = 16000  # サンプリングレート
    duration = 2  # 秒
    t = np.linspace(0, duration, int(fs*duration), endpoint=False)
    speaker_signal = np.sin(2 * np.pi * 100 * t)  # スピーカーからの100Hzのサイン波
    mic_signal = speaker_signal + 0.5 * np.roll(speaker_signal, 100)  # 遅延と減衰を加えたエコーを含むマイク信号

    # エコーキャンセルの適用
    filtered_signal, error_signal = lms_filter(mic_signal, speaker_signal)

    # 結果のプロット（matplotlibが必要です）
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(t, mic_signal, label='Original Mic Signal')
    plt.plot(t, filtered_signal, label='Filtered Signal')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    #mlstest()
    main()