import sys,os
import time
import traceback
from threading import Lock

import numpy as np


sys.path.append(os.getcwd())
import pyaudio  # PyAudioを使用 (DirectSoundの代替としてクロスプラットフォームの音声処理ライブラリ)

from BotVoice.rec_util import AudioI8, AudioI16, AudioF32, EmptyF32, np_append, save_wave, load_wave, signal_ave, sin_signal

class DirectSoundEchoCanceller:
    def __init__(self, pa_chunk_size, delay_ms, sample_rate):
        self._lock = Lock()
        self.pa_chunk_size = pa_chunk_size
        self.delay_samples = int((delay_ms / 1000) * sample_rate)
        self.sample_rate = sample_rate
        #
        self.pyaudio = pyaudio.PyAudio()
        self.stream = None
        # 再生データ
        self.play_f32:AudioF32 = EmptyF32
        self.play_pos:int = 0
        self.is_playing = False
        self.next_echo:AudioF32 = np.zeros( self.pa_chunk_size, dtype=np.float32 )
        # 録音データ
        self.mic_buffer:list[AudioF32] = []
        self.echo_buffer:list[AudioF32] = []

        # パラメータ設定
        self.mu = 0.001  # 学習率
        self.filter_order = 2000 # 128  # フィルターの長さ

        # エコーキャンセルフィルター
        self.echo_filter = np.zeros(self.filter_order)

    def start(self):
        """録音・再生を同時に開始"""
        print("start ",end="")
        self.stream = self.pyaudio.open( input_device_index=15,
            format=pyaudio.paFloat32,
                                                channels=1,
                                                rate=self.sample_rate,
                                                input=True,
                                                output=True,
                                                frames_per_buffer=self.pa_chunk_size,
                                                stream_callback=self.callback)
        self.stream.start_stream()
        print("----------")
        print("----------")

    def stop(self):
        """ストリームを停止"""
        print("stop ",end="")
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.pyaudio.terminate()

    def play(self, playback_data:AudioF32|None ):
        """再生データを設定し、再生を開始"""
        print("play ",end="")
        with self._lock:
            self.play_f32 = playback_data if playback_data is not None else EmptyF32
            self.play_pos = 0
            self.is_playing = True if len(self.play_f32)>0 else False

    def callback(self, in_data, frame_count, time_info, status):
        out_f32:AudioF32 = np.zeros( frame_count, dtype=np.float32 )
        try:
            """録音と再生をコールバックで処理"""
            # マイクデータ取得
            mic_data = np.frombuffer(in_data, dtype=np.float32)
            with self._lock:
                self.mic_buffer.append(mic_data)
                self.echo_buffer.append(self.next_echo)
                n:int = min( frame_count, len(self.play_f32)-self.play_pos)
                if n>0:
                    # 再生データが設定されている場合は再生
                    out_f32[0:n] = self.play_f32[self.play_pos:self.play_pos+n]
                    self.play_pos+=n
                elif self.is_playing:
                    print(f"<EOP>")
                    self.play_f32 = EmptyF32
                    self.play_pos=0
                    self.is_playing = False

            self.next_echo = out_f32
            out_data = out_f32.tobytes()
            out_data = out_f32.tobytes()
            return (out_data, pyaudio.paContinue)
        except:
            traceback.print_exc()
        self.play_f32 = EmptyF32
        self.play_pos=0
        self.is_playing = False
        out_data = out_f32.tobytes()
        return (out_data, pyaudio.paAbort)

    def apply_delay_compensation(self):
        """遅延補正を行う"""
        with self._lock:
            a:list[AudioF32]=self.mic_buffer
            self.mic_buffer = []
            b:list[AudioF32]=self.echo_buffer
            self.echo_buffer = []
        if len(a)==0:
            return EmptyF32
        mic_f32:AudioF32 = np.concatenate( a )
        echo_f32:AudioF32 = np.concatenate( b )
        print(f"cancel {len(a)} : {len(b)} , {len(mic_f32)} : {len(echo_f32)}")

        offset_mic_f32 = mic_f32[self.delay_samples:]
        offset_echo_f32 = echo_f32[:len(offset_mic_f32)]

        filterd_f32 = self.lms_echo_cancel( offset_mic_f32, offset_echo_f32 )
        return filterd_f32,mic_f32
        # compensated_mic = np.zeros_like(self.mic_buffer)
        # if self.delay_samples < len(self.mic_buffer):
        #     compensated_mic[self.delay_samples:] = self.mic_buffer[:-self.delay_samples]
        # return compensated_mic

    # LMSエコーキャンセラー関数
    def lms_echo_cancel(self,input_signal:AudioF32, echo_signal:AudioF32) ->AudioF32:
        print(f"[LMS] input {input_signal.shape} {input_signal.dtype} {min(input_signal)} to {max(input_signal)}")
        print(f"[LMS] echo  {echo_signal.shape} {echo_signal.dtype} {min(echo_signal)} to {max(echo_signal)}")
        output = np.zeros_like(input_signal)
        for n in range(len(input_signal) - self.filter_order):
            # フィルターの適用
            echo_estimate = np.dot(self.echo_filter, echo_signal[n:n + self.filter_order])
            try:
                output[n] = input_signal[n] - echo_estimate
            except:
                print("error")
            
            # 誤差計算
            error = input_signal[n] - echo_estimate
            
            # フィルター更新
            self.echo_filter += 2 * self.mu * error * echo_signal[n:n + self.filter_order]
        return output

    def get_audio(self):
        """エコーキャンセルした音声データを取得"""
        return self.apply_delay_compensation()
        # compensated_mic = self.apply_delay_compensation()
        # stereo_output = np.vstack((self.playback_buffer, compensated_mic)).T
        # return stereo_output

def list_microphones():
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    num_devices = info.get('deviceCount')

    print("Available Microphones and their Specifications:")
    for i in range(num_devices):
        device_info = p.get_device_info_by_host_api_device_index(0, i)
        if device_info.get('maxInputChannels') > 0:
            print(f"\nDevice ID: {i}")
            print(f"Name: {device_info.get('name')}")
            print(f"Max Input Channels: {device_info.get('maxInputChannels')}")
            print(f"Default Sample Rate: {device_info.get('defaultSampleRate')}")
            print(f"Host API: {device_info.get('hostApi')}")
            print(f"Input Latency: {device_info.get('defaultLowInputLatency')}")

    p.terminate()

def main():
    list_microphones()
    # 使用例
    pa_chunk_size = 3200
    delay_ms = 10
    sample_rate:int = 16000

    nsec = 10.0  # 例えば、2秒の音を生成したい場合

    canceller:DirectSoundEchoCanceller = DirectSoundEchoCanceller(pa_chunk_size, delay_ms, sample_rate)
    canceller.start()

    # 再生音を設定 (例えば、440Hzのサイン波)
    t = np.arange(int(nsec * sample_rate))  # サンプル数を計算してnp.arangeの範囲を指定
    playback_data = np.sin(2 * np.pi * 440 * t / sample_rate).astype(np.float32)
    playback_data7 = playback_data * 0.1

    playback_data7:AudioF32 = load_wave('test/testData/sample_voice.wav',sampling_rate=sample_rate )

    canceller.play(playback_data7)
    time.sleep(nsec)
    while canceller.is_playing:
        print("+",end="")
        time.sleep(0.5)

    # ストリームが動作中にエコーキャンセル後の音声データを取得
    output,raw = canceller.get_audio()

    # 停止
    canceller.stop()
    print(f"[OUT] input {output.shape} {output.dtype} {min(output)} to {max(output)}")
    print(f"[OUT] input {raw.shape} {raw.dtype} {min(raw)} to {max(raw)}")

    save_wave( 'tmp/lms_output.wav', output, sampling_rate=sample_rate, ch=1)
    save_wave( 'tmp/raw_output.wav', raw, sampling_rate=sample_rate, ch=1)

if __name__ == "__main__":
    main()