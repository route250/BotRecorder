import sys,os
import time
from threading import Lock

import pyaudio
from pyaudio import Stream
import wave
import numpy as np

import matplotlib.pyplot as plt
sys.path.append(os.getcwd())
from BotVoice.rec_util import AudioF32, np_append, save_wave, load_wave, signal_ave, sin_signal

# 録音/再生の設定
PA_FORMAT = pyaudio.paFloat32
CHANNELS:int = 1
RATE:int = 16000  # 16kHz
CHUNK_SEC:float = 0.2 # 0.2sec
CHUNK_LEN:int = int(RATE*CHUNK_SEC)
BUFFER_LEN:int = int(RATE/CHUNK_LEN) * CHUNK_LEN

import numpy as np

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
        self._paudio:pyaudio.PyAudio|None = pyaudio.PyAudio()
        self._stream:Stream|None = None
        # 録音設定
        self._rec_boost:float = 3.0

        # 再生用
        self._play_byffer_list:list[AudioF32] = []
        self._play_buffer:AudioF32|None = None
        self._play_pos:int = 0
        # エコーバックデータ保存用
        self._echo_list:list[AudioF32] = [EchoLessRecorder.ZerosF32,EchoLessRecorder.ZerosF32]
        # 録音データの保存用
        self._rec_list:list[AudioF32] = []
        # 
        self._echo_delay_sec:float = 0.0
        self._echo_delay_idx:int = 0
        #
        self.echo_cancel:bool = True

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
            return self._stream and self._stream and self._stream.is_active()
    
    def is_stopped(self):
        with self._lock:
            return self._stream is None or self._stream.is_stopped()

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
                st.close()
        except:
            pass
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
        if time_info:
            rt:float = time_info.get('input_buffer_adc_time')
            pt:float = time_info.get('output_buffer_dac_time')
            if rt and pt:
                delay_sec = pt - rt
                delay_idx = int( RATE*delay_sec )
                if self._echo_delay_sec != delay_sec or self._echo_delay_idx != delay_idx:
                    self._echo_delay_sec = delay_sec
                    self._echo_delay_idx = delay_idx
                    print(f"time_info: {delay_idx}(fr) {delay_sec:.6f}(sec)")
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
            self._echo_list.append( play_f32 )

        return (play_f32.tobytes(),pyaudio.paContinue)

    def get_audio(self) ->tuple[AudioF32,AudioF32,AudioF32]:
        delayc:int = 2
        with self._lock:
            es:int = delayc*3
            if len(self._rec_list)<es:
                return EchoLessRecorder.EmptyF32,EchoLessRecorder.EmptyF32,EchoLessRecorder.EmptyF32
            # 録音データをnumpy配列に変換
            rec_buf = self._rec_list
            self._rec_list = []
            echo_buf = self._echo_list
            self._echo_list = self._echo_list[-es:]
        # 録音データ
        raw_f32:AudioF32 = np.concatenate(rec_buf)
        if 0.0<self._rec_boost<=10.0:
            raw_f32 = raw_f32 * self._rec_boost
        # 音響データ
        echo_full_f32:AudioF32 = np.concatenate(echo_buf)
        # フィルタ処理
        filtered_f32, echo_f32 = self._apply_filter( raw_f32, echo_full_f32 )

        return filtered_f32,raw_f32,echo_f32

    def _apply_filter(self, raw_audio_f32:AudioF32, raw_echo_audio_f32:AudioF32) ->tuple[AudioF32,AudioF32]:
        raw_len:int = raw_audio_f32.shape[0]
        offset_end:int = raw_echo_audio_f32.shape[0]-raw_len
        if offset_end<0:
            raise ValueError('invalid buffer size???')
        offset_center:int = max(0,offset_end - self._echo_delay_idx)

        echo_ave:float = signal_ave(raw_echo_audio_f32)
        raw_ave:float = signal_ave(raw_audio_f32)
        if echo_ave>0.001 and raw_ave>0.001:
            sig_rate:float = raw_ave / echo_ave
            echo_audio_f32 = raw_echo_audio_f32 * sig_rate
        else:
            echo_audio_f32 = raw_echo_audio_f32

        if not self.echo_cancel or echo_ave<=0.001 or raw_ave<=0.001:
            # diff 5423
            offset = offset_center+390
            echod = echo_audio_f32[offset:offset+raw_len]
            filterd = raw_audio_f32 - echod
            return filterd, echod
 
        # 移動平均のウィンドウサイズ
        window_size = 5
        # 平均化のためのカーネルを作成
        window_kernel = np.ones(window_size) / window_size
        # 移動平均を計算
        moving_average = np.convolve(echo_audio_f32, window_kernel, mode='same')

        # 再生音を引き算してノイズ除去
        filtered_audio_f32:AudioF32 = raw_audio_f32
        best_volume:float = sys.float_info.max
        best_offset:int = 0
        best_lv:float = 1.0
        best_mask = raw_audio_f32

        st = max(0, offset_center +380 )
        ed = min(offset_end, offset_center + 400 )
        # 最適なオフセットを探す
        for offset in range(st,ed):
            mask_f32 = moving_average[offset:offset+raw_len]
            subtracted_f32 = raw_audio_f32 - mask_f32
            volume:float = np.sum(np.abs(subtracted_f32))
            if volume < best_volume:
                filtered_audio_f32 = subtracted_f32
                best_volume = volume
                best_offset = offset
                best_mask = mask_f32

        lo_rate = 0.5
        hi_rate = 1.5
        lv = lo_rate
        while lv<=hi_rate:
            mask_f32 = best_mask *lv
            subtracted_f32 = raw_audio_f32 - mask_f32
            volume:float = np.sum(np.abs(subtracted_f32))
            if volume < best_volume:
                filtered_audio_f32 = subtracted_f32
                best_volume = volume
                best_lv = lv
            lv += 0.1
        best_mask = best_mask * best_lv
        best_echo = echo_audio_f32[offset:offset+raw_len] * best_lv
        print(f"filter offset:{best_offset:6d} rate:{best_lv:.3f} vol:{best_volume:.2f}")
        return filtered_audio_f32, best_echo

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

def main():

    # 読み込むWaveファイルの設定
    play_filename = 'test/testData/ttsmaker-file-2024-8-27-20-36-50.wav'
    output_raw_filename = 'tmp/raw_audio.wav'
    output_filtered_filename = 'tmp/filtered_audio.wav'
    output_echo_filename = 'tmp/echo_audio.wav'

    el_recorder:EchoLessRecorder = EchoLessRecorder()
    el_recorder.echo_cancel = True

    # 再生音をnumpy配列に読み込む
    play_audio_f32:AudioF32 = load_wave( play_filename, sampling_rate=RATE )
    # play_audio_f32 = sin_signal( freq=220, duration=3.0, vol=0.5 )
    
    el_recorder.add_play( play_audio_f32 )


    filterd_audio_list = []
    raw_audio_list = []
    echo_audio_list = []
    # 指定した期間録音
    print("録音と再生を開始します...")
    el_recorder.start()
    while el_recorder.is_active() and el_recorder.is_playing():
        filtered_seg_f32, raw_seg_f32, echo_seg_f32 = el_recorder.get_audio()
        if len(filtered_seg_f32)>0:
            filterd_audio_list.append(filtered_seg_f32)
            raw_audio_list.append(raw_seg_f32)
            echo_audio_list.append(echo_seg_f32)
        time.sleep( 1.5 )  # ミリ秒単位で指定
    print("終了しました")
    el_recorder.stop()

    # 生の録音音声を保存
    raw_seg_f32 = np.concatenate( raw_audio_list )
    save_wave(output_raw_filename, raw_seg_f32, sampling_rate=RATE,ch=1)
    print(f"録音されたデータが {output_raw_filename} に保存されました。")

    # 再生音声を保存
    echo_audio_f32 = np.concatenate(echo_audio_list )
    save_wave(output_echo_filename, echo_audio_f32, sampling_rate=RATE,ch=1)
    print(f"再生データが {output_echo_filename} に保存されました。")

    # 引き算後の音声を保存
    filtered_audio_f32 = np.concatenate(filterd_audio_list )
    save_wave(output_filtered_filename, filtered_audio_f32, sampling_rate=RATE,ch=1)
    print(f"引き算されたデータが {output_filtered_filename} に保存されました。")

    ps:int = CHUNK_LEN*5
    pe:int = CHUNK_LEN*6

    plt.figure()

    plt.plot(raw_seg_f32[ps:pe], label='Mic Signal')
    plt.plot(filtered_audio_f32[ps:pe], label='Filtered Signal', alpha=0.5)
    plt.legend()

    plt.figure()
    plt.plot(raw_seg_f32[ps:pe], label='Mic Signal')
    plt.plot(echo_audio_f32[ps:pe], label='Echo Signal', alpha=0.5)
    plt.legend()

    plt.show()

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
    plt.figure()
    plt.plot(t, mic_signal[CHUNK_LEN:CHUNK_LEN*3], label='Original Mic Signal')
    plt.plot(t, filtered_signal[CHUNK_LEN:CHUNK_LEN*3], label='Filtered Signal')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    #mlstest()
    main()