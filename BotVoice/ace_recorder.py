import sys,os
import time
import traceback
from threading import Lock
from typing import NamedTuple
import math
import audioop

import numpy as np
import sounddevice as sd


sys.path.append(os.getcwd())
from BotVoice.rec_util import AudioI16, AudioF32, EmptyF32, np_append, save_wave, load_wave, signal_ave, sin_signal
from BotVoice.rec_util import f32_to_i16, i16_to_f32, to_f32, resample, generate_mixed_tone, audio_info

def maek_marker_tone( size:int, sample_rate:int,freq1:int,freq2:int, vol:float=0.9):
    tone_sec = size/2/sample_rate
    tone_f32:AudioF32 = generate_mixed_tone( tone_sec, freq1, freq2, sample_rate, vol=vol )
    tone_all:AudioF32 = np.zeros( size, dtype=np.float32 )
    s:int = 0
    tone_all[s:s+len(tone_f32)] = tone_f32
    return tone_all

def nlms_echo_cancel(mic: np.ndarray, spk_f32: np.ndarray, mu: float, w: np.ndarray) -> np.ndarray:
    """
    LMSアルゴリズムによるエコーキャンセルを実行する関数。

    Parameters:
    mic (np.ndarray): マイクからの入力信号
    spk (np.ndarray): スピーカーからの出力信号
    mu (float): ステップサイズ（学習率）
    w (np.ndarray): フィルタ係数ベクトルの初期値

    Returns:
    np.ndarray: エコーキャンセル後の信号
    """
    if not isinstance(mic,np.ndarray) or len(mic.shape)!=1 or mic.dtype!=np.float32:
        raise TypeError()
    if not isinstance(spk_f32,np.ndarray) or len(spk_f32.shape)!=1 or spk_f32.dtype!=np.float32:
        raise TypeError()
    if not isinstance(w,np.ndarray) or len(w.shape)!=1 or w.dtype!=np.float64:
        raise TypeError()
    if len(mic) != len(spk_f32)-len(w):
        raise TypeError("invalid array length")
    if np.isnan(mic).any() or np.isinf(mic).any():
        raise ValueError("mic include NaN or INF")
    if np.isnan(spk_f32).any() or np.isinf(spk_f32).any():
        raise ValueError("spk include NaN or INF")
    mic_len = len(mic)
    num_taps = len(w)

    # エコーキャンセル後の信号を保存する配列
    cancelled_signal = np.zeros(mic_len,dtype=np.float32)

    spk_f64 = spk_f32.astype(np.float64)

    AEC_PLIMIT:float = 1e30
    maxlv = np.sum(np.abs(w))
    if maxlv>AEC_PLIMIT:
        print(f"[WARN] w is too large {maxlv}")
        w *= (AEC_PLIMIT/maxlv)

    # スピーカー出力の全項目の二乗を事前に計算
    spk_squared = spk_f64 ** 2
    # 有効な範囲内でのみ計算を実行
    factor = np.zeros(mic_len,dtype=np.float64)
    for n in range(mic_len):
        factor[n] = np.sum(spk_squared[n:n+num_taps])

    # LMSアルゴリズムのメインループ
    for mu3 in (mu,):
        for n in range(mic_len):
                # スピーカー出力 spk の一部をスライスして使う (直近の num_taps サンプルを使う)
                spk_slice = spk_f64[n:n+num_taps]  # スライスしてタップ分の信号を取得
                
                # スピーカー出力 spk_slice とフィルタ係数 w の内積によるフィルタ出力 y(n) を計算
                y = np.dot(w, spk_slice)
                # if np.isnan(y) or np.isinf(y):
                #     print("[ERR] t is NaN or INF")
                #     cancelled_signal[n] = mic[n]
                #     continue
                
                # エラー e(n) を計算 (マイク信号 mic[n] とフィルタ出力 y の差)
                e = mic[n] - y

                # エコーキャンセル後の信号を計算 (マイク信号から予測されたエコーを引く)
                cancelled_signal[n] = e
                # フィルタ係数の更新式
                norm_factor = factor[n] # np.dot(spk_slice, spk_slice)
                # if norm_factor != np.dot(spk_slice, spk_slice):
                #     print(f"ERROR:invalid norm_factor")
                if norm_factor >1e-1:
                    w[:] = w + (e*mu3/norm_factor) * spk_slice
                    # if np.isnan(w).any() or np.isinf(w).any():
                    #     raise ValueError("spk include NaN or INF")

    return np.clip(cancelled_signal,-0.99,0.99)

class SpkPair(NamedTuple):
    f32:AudioF32
    i16:AudioI16

class AecRecorder:

    H1:int=440
    H2:int=880

    def __init__(self, device=None, pa_chunk_size:int=3200, sample_rate:int=16000):
        self._lock = Lock()
        self.device = device
        self.ds_chunk_size = pa_chunk_size
        self.sample_rate = sample_rate
        #
        self._stream = None
        self._callback_cnt:int = 0

        #
        self.zeros_f32 = np.zeros( self.ds_chunk_size, dtype=np.float32 )
        self.zeros_i16 = np.zeros( self.ds_chunk_size, dtype=np.int16 )

        # 再生データ
        self.play_data:list[SpkPair] = []
        self.play_pos:int = 0
        self._is_playing = False
        self._post_play_count:int = 0
        self._post_play_num:int = 0

        # 録音データ
        self.mic_boost:float = 1.0
        self.mic_buffer:list[AudioI16] = []
        self._detect_num:int = int( 2 * self.sample_rate / self.ds_chunk_size )
        self.spk_buffer:list[AudioF32] = [self.zeros_f32] * self._detect_num

        # 先頭マーカー検出
        self._marker_lv:float = 0.4
        tone1:AudioF32 = maek_marker_tone( self.ds_chunk_size, sample_rate, AecRecorder.H1, AecRecorder.H2, vol=self._marker_lv )
        self.marker_tone_f32:AudioF32 = tone1
        self.marker_tone_I16:AudioI16 = f32_to_i16( self.marker_tone_f32 )
        self.marker_bytes:bytes = f32_to_i16( tone1 ).tobytes()
        self.marker_pair:SpkPair = SpkPair( self.marker_tone_f32, self.marker_tone_I16 )
        self.zeros_pair:SpkPair = SpkPair( self.zeros_f32, self.zeros_i16 )
        self._detectbuf:bytes = b''
        self._detect_cnt:int = -1
        self._before_pos:int = -1
        self.delay_samples:int = 0

        # エコーキャンセルフィルター
        self.aec_mu = 0.2 # 学習率
        self.aec_taps = 700 # フィルターの長さ
        self.aec_offset = -100 # 先頭がよくずれるので余裕を
        peek = max(self.aec_taps, self.aec_taps + self.aec_offset)-2
        if peek<self.aec_taps:
            w1= np.linspace(0.0,0.5,peek,dtype=np.float64)
            w2= np.linspace(0.2,0.0,self.aec_taps-peek,dtype=np.float64)
            self.aec_w = np.concatenate( (w1,w2))
        else:
            self.aec_w = np.linspace(0.0,0.5,self.aec_taps,dtype=np.float64)
        AEC_PLIMIT:float = 1.0
        maxlv = np.sum(np.abs(self.aec_w))
        if maxlv>AEC_PLIMIT:
            self.aec_w *= (AEC_PLIMIT/maxlv)
        self.aec_plimit:float = 1.2

    def is_playing(self) ->int:
        with self._lock:
            if self._is_playing:
                return 1
        return 0

    def is_active(self) ->bool:
        with self._lock:
            if self._stream:
                return self._stream.active
            else:
                print("not open?")
                return False
    
    def is_stopped(self):
        with self._lock:
            if self._stream:
                return self._stream.closed
            else:
                return True

    def start(self):
        """録音・再生を同時に開始"""
        print("start ",end="")
        self._stream = sd.Stream( samplerate=self.sample_rate,
                                blocksize=self.ds_chunk_size,
                                device = self.device, channels=1, dtype=np.int16, callback=self._callback )
        self._stream.start()
        print("----------")
        print("----------")

    def stop(self):
        """ストリームを停止"""
        print("stop ",end="")
        if self._stream:
            self._stream.stop(ignore_errors=True)

    def play_marker(self):
        """再生データを設定し、再生を開始"""
        with self._lock:
            if len(self.play_data)>0:
                self.play_data.append( self.zeros_pair )
            self.play_data.append( self.marker_pair )
            self.play_data.append( self.zeros_pair )
            self._is_playing = True

    def play(self, audio:AudioF32|None, sr:int|None=None ):
        """再生データを設定し、再生を開始"""
        print("play ",end="")
        audio_f32:AudioF32 = to_f32(audio)
        if audio_f32 is not None and len(audio_f32)>0:
            if isinstance(sr,int|float) and sr>0:
                audio_f32 = resample(audio_f32,orig=sr,target=self.sample_rate)
            max_lv = np.max(np.abs(audio_f32))
            if max_lv>0.3:
                audio_f32 *= (0.3/max_lv)
            padding_f32:AudioF32 = np.zeros( self.ds_chunk_size, dtype=np.float32 )
            play_f32:AudioF32 = np.concatenate( (audio_f32,padding_f32) )
            play_i16:AudioI16 = f32_to_i16(play_f32)
            size:int = len(audio_f32)
            step:int = self.ds_chunk_size
            data:list[SpkPair] = [ SpkPair(play_f32[s:s+step],play_i16[s:s+step]) for s in range(0,size,step) ]
            with self._lock:
                self.play_data.extend( data )
                self._is_playing = True

    def _callback(self, inbytes:np.ndarray, outdata:np.ndarray, frames:int, time, status ) ->None:
        try:
            if frames != self.ds_chunk_size:
                print(f" invalid callback frames {frames} {status}")
            mic_data:AudioI16 = inbytes[:,0].copy()
            with self._lock:
                self.mic_buffer.append( mic_data )
                if 0<=self._detect_cnt:
                    if self._detect_cnt<self._detect_num:
                        self._detectbuf += mic_data.tobytes()
                        pos,factor = audioop.findfit( self._detectbuf, self.marker_bytes ) if self._detect_cnt>2 else (0,0.0)
                        if self._detect_cnt>5 and 0<=pos and pos==self._before_pos:
                            tmp = self._detectbuf[pos*2:pos*2+self.ds_chunk_size]
                            lo,hi = audioop.minmax( tmp, 2 )
                            maxlv = (abs(lo)+abs(hi))/2/32768
                            self.mic_boost = self._marker_lv/maxlv
                            delay:int = pos + self.ds_chunk_size
                            print(f"[SND]delay: pos:{pos} OK {delay} factor:{factor} maxlv:{maxlv} boost:{self.mic_boost}")
                            self.delay_samples = delay
                            self._detect_cnt=-1
                            self._detectbuf=b''
                        else:
                            print(f"[SND]delay: pos:{pos}")
                            self._detect_cnt+=1
                            self._before_pos = pos
                    else:
                        self._detect_cnt = -1
                        self.delay_samples = 0
                        self._detectbuf=b''
                        print(f"[SND]delay:NotFound")

                pax = self.play_data
                if len(pax)>0:
                    self._post_play_count = self._post_play_num
                    # 再生データが設定されている場合は再生
                    play = pax.pop(0)
                    if play is self.marker_pair:
                        self._detect_cnt = 0
                        self._before_pos = -1
                        self._detectbuf = b''
                        print(f"[SND]delay:Start")
                else:
                    if self._post_play_count>0:
                        self._post_play_count-=1
                    else:
                        self._is_playing = False
                    play = self.zeros_pair
                self.spk_buffer.append( play.f32 )

            outdata[:,0] = play.i16[:]
        except:
            traceback.print_exc()
        finally:
            self._callback_cnt+=1

    def copy_raw_buffer(self) ->tuple[AudioF32,AudioF32]:
        with self._lock:
            e:list[AudioF32]=self.spk_buffer.copy()
            m:list[AudioI16]=self.mic_buffer.copy()
        mic_f32:AudioF32 = i16_to_f32( np.concatenate( m ) )
        spk_f32:AudioF32 = np.concatenate( e )[-len(mic_f32):]
        return mic_f32,spk_f32

    def copy_raw_audio(self) ->tuple[AudioF32,AudioF32]:
        with self._lock:
            mic_buf:list[AudioI16]=self.mic_buffer.copy()
            delay_samples = max(0,self.delay_samples+self.aec_offset)
            x:int = int( (delay_samples+len(self.aec_w))/self.ds_chunk_size ) + 1
            spk_buf: list[AudioF32] = self.spk_buffer[-len(mic_buf)-x:]
        mic_f32:AudioF32 = i16_to_f32( np.concatenate( mic_buf ) ) * self.mic_boost
        spk_f32:AudioF32 = np.concatenate( spk_buf )
        spk_f32 = spk_f32[-len(mic_f32)-delay_samples-len(self.aec_w):len(spk_f32)-delay_samples]
        return mic_f32,spk_f32

    def get_raw_audio(self,keep:bool=False) ->tuple[AudioF32,AudioF32]:
        with self._lock:
            if self._detect_cnt>=0:
                return EmptyF32,EmptyF32
            if keep:
                mic_buf:list[AudioI16]=self.mic_buffer.copy()
            else:
                mic_buf:list[AudioI16]=self.mic_buffer
                self.mic_buffer = []

            delay_samples = max(0,self.delay_samples+self.aec_offset)
            x:int = int( (delay_samples+len(self.aec_w))/self.ds_chunk_size ) + 1
            spk_buf: list[AudioF32] = self.spk_buffer[-len(mic_buf)-x:].copy()
            if not keep:
                self.spk_buffer = self.spk_buffer[-self._detect_num:]

        if len(mic_buf)==0:
            return EmptyF32, EmptyF32
        mic_f32:AudioF32 = i16_to_f32( np.concatenate( mic_buf ) ) * self.mic_boost
        spk_f32:AudioF32 = np.concatenate( spk_buf )
        spk_f32 = spk_f32[-len(mic_f32)-delay_samples-len(self.aec_w):len(spk_f32)-delay_samples]
        return mic_f32,spk_f32
    
    def get_aec_audio(self) ->tuple[AudioF32,AudioF32,AudioF32]:
        mic_f32, spk_f32 = self.get_raw_audio()
        if len(mic_f32)==0:
            return mic_f32,mic_f32,mic_f32
        lms_f32:AudioF32 = nlms_echo_cancel( mic_f32, spk_f32, self.aec_mu, self.aec_w )
        # lms_f32:AudioF32 = rls_echo_cancel( mic_f32, spk_f32, 0.98, 100, self.aec_w, self.aec_offset )
        return lms_f32, mic_f32, spk_f32

    def get_audio(self) ->AudioF32:
        lms_f32,_,_ = self.get_aec_audio()
        return lms_f32

def list_microphones():
    print( sd.query_devices() )
#     p = pyaudio.PyAudio()
#     info = p.get_host_api_info_by_index(0)
#     num_devices = as_int( info.get('deviceCount'), 0 )

#     print("Available Microphones and their Specifications:")
#     for i in range(num_devices):
#         device_info = p.get_device_info_by_host_api_device_index(0, i)
#         inch = as_int( device_info.get('maxInputChannels') ) 
#         if inch > 0:
#             print(f"\nDevice ID: {i}")
#             print(f"Name: {device_info.get('name')}")
#             print(f"Max Input Channels: {device_info.get('maxInputChannels')}")
#             print(f"Default Sample Rate: {device_info.get('defaultSampleRate')}")
#             print(f"Host API: {device_info.get('hostApi')}")
#             print(f"Input Latency: {device_info.get('defaultLowInputLatency')}")

#     p.terminate()

