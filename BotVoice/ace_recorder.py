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

def lms_echo_cancel(mic: np.ndarray, spk: np.ndarray, mu: float, w: np.ndarray, delay:int=0) -> np.ndarray:
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
    if not isinstance(spk,np.ndarray) or len(spk.shape)!=1 or spk.dtype!=np.float32:
        raise TypeError()
    if not isinstance(w,np.ndarray) or len(w.shape)!=1 or w.dtype!=np.float32:
        raise TypeError()
    if len(mic) != len(spk):
        raise TypeError()
    
    mic_len = len(mic)
    num_taps = len(w)

    shift:int = delay + num_taps

    # エコーキャンセル後の信号を保存する配列
    cancelled_signal = np.zeros(mic_len,dtype=np.float32)

    AEC_PLIMIT:float = 1.0
    maxlv = np.sum(np.abs(w))
    if maxlv>AEC_PLIMIT:
        print(f"[WARN] w is too large {maxlv}")
        w *= (AEC_PLIMIT/maxlv)

    # LMSアルゴリズムのメインループ
    for n in range(mic_len):
        xs = n-shift
        xe = xs+num_taps
        if xs<0 or len(spk)<xe:
            cancelled_signal[n] = mic[n]
        else:
            # スピーカー出力 spk の一部をスライスして使う (直近の num_taps サンプルを使う)
            spk_slice = spk[n-shift:n-shift+num_taps]  # スライスしてタップ分の信号を取得
            
            # スピーカー出力 spk_slice とフィルタ係数 w の内積によるフィルタ出力 y(n) を計算
            y = np.dot(w, spk_slice)
            if np.isnan(y) or np.isinf(y):
                print("[ERR] t is NaN or INF")
                cancelled_signal[n] = mic[n]
                continue
            abs_y = np.abs(y)
            if abs_y>1:
                # print(f"[WARN] abs_y is over? {abs_y}")
                w *= 0.9
                y *= 0.9
            
            # エラー e(n) を計算 (マイク信号 mic[n] とフィルタ出力 y の差)
            e = mic[n] - y

            # エコーキャンセル後の信号を計算 (マイク信号から予測されたエコーを引く)
            cancelled_signal[n] = np.clip(e, -1.0, 1.0 )
            
            # フィルタ係数の更新式
            zr:float = np.count_nonzero(spk_slice)/len(spk_slice)
            if zr>0.9:
                w[:] = w + mu * e * spk_slice
                maxlv = np.sum(np.abs(w))
                if maxlv>AEC_PLIMIT:
                    # print(f"[WARN] w is too large {maxlv}")
                    w *= (AEC_PLIMIT/maxlv)
    return cancelled_signal


def nlms_echo_cancel(mic: np.ndarray, spk: np.ndarray, mu: float, w: np.ndarray, delay: int = 0, epsilon: float = 1e-6) -> np.ndarray:
    """
    NLMSアルゴリズムによるエコーキャンセルを実行する関数。

    Parameters:
    mic (np.ndarray): マイクからの入力信号（一次元の np.float32 型配列）
    spk (np.ndarray): スピーカーからの出力信号（一次元の np.float32 型配列）
    mu (float): ステップサイズ（学習率）。通常は0 < mu <= 1
    w (np.ndarray): フィルタ係数ベクトルの初期値（一次元の np.float32 型配列）
    delay (int): 遅延時間（サンプル数）
    epsilon (float): 零割り防止のための微小値

    Returns:
    np.ndarray: エコーキャンセル後の信号
    """
    # 入力の型と形状をチェック
    if not isinstance(mic, np.ndarray) or mic.ndim != 1 or mic.dtype != np.float32:
        raise ValueError("mic は一次元の np.float32 型の ndarray である必要があります。")
    if not isinstance(spk, np.ndarray) or spk.ndim != 1 or spk.dtype != np.float32:
        raise ValueError("spk は一次元の np.float32 型の ndarray である必要があります。")
    if not isinstance(w, np.ndarray) or w.ndim != 1 or w.dtype != np.float32:
        raise ValueError("w は一次元の np.float32 型の ndarray である必要があります。")
    if min(mic)<-1.0 or 1.0<max(mic):
        raise ValueError("mic の値が範囲外です。")
    if min(spk)<-1.0 or 1.0<max(spk):
        raise ValueError("mic の値が範囲外です。")
    mic_len = len(mic)
    num_taps = len(w)

    shift: int = delay + num_taps

    # エコーキャンセル後の信号を保存する配列
    cancelled_signal = np.zeros(mic_len, dtype=np.float32)

    # NLMSアルゴリズムのメインループ
    for n in range(mic_len):
        xs = n - shift
        xe = xs + num_taps
        if xs < 0 or xe > len(spk):
            cancelled_signal[n] = mic[n]
        else:
            # スピーカー出力 spk の一部をスライスして使う
            spk_slice = spk[xs:xe]

            # フィルタ出力 y(n) を計算
            r = 1.0
            y = np.dot(w, spk_slice)
            # if y<-1.0 or 1.0<y:
            #     r = 1.0/abs(y)
            #     y *= r
            # エラー e(n) を計算
            e = mic[n] - y

            # エコーキャンセル後の信号を保存
            cancelled_signal[n] = e

            zr:float = np.count_nonzero(spk_slice)/len(spk_slice)
            if zr>0.9:
                # ステップサイズの正規化項を計算
                norm_factor = np.dot(spk_slice, spk_slice)
                if norm_factor>epsilon:
                    # フィルタ係数の更新式（NLMS）
                    w[:] = w*r + (mu / norm_factor) * e * spk_slice

    return cancelled_signal

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
        self.stream = None
        self._callback_cnt:int = 0

        #
        self.zeros_f32 = np.zeros( self.ds_chunk_size, dtype=np.float32 )
        self.zeros_i16 = np.zeros( self.ds_chunk_size, dtype=np.int16 )

        # 再生データ
        self.play_data:list[SpkPair] = []
        self.play_pos:int = 0
        self.is_playing = False
        self._post_play_count:int = 0
        self._post_play_num:int = 0

        # 録音データ
        self.mic_boost:float = 1.0
        self.mic_buffer:list[AudioI16] = []
        self.echo_buffer:list[AudioF32] = []

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
        self._detect_num:int = int( 2 * self.sample_rate / self.ds_chunk_size )
        self._before_pos:int = -1
        self.delay_samples:int = 0
        self.delay_factor:float = 0.0

        # エコーキャンセルフィルター
        self.aec_mu = 0.1  # 学習率
        self.aec_taps = 1500 # フィルターの長さ
        self.aec_offset = -200 # 先頭がよくずれるので余裕を
        peek = max(self.aec_taps, self.aec_taps + self.aec_offset)-2
        if peek<self.aec_taps:
            w1= np.linspace(0.0,0.5,peek,dtype=np.float32)
            w2= np.linspace(0.2,0.0,self.aec_taps-peek,dtype=np.float32)
            self.aec_w = np.concatenate( (w1,w2))
        else:
            self.aec_w = np.linspace(0.0,0.5,self.aec_taps,dtype=np.float32)
        AEC_PLIMIT:float = 1.0
        maxlv = np.sum(np.abs(self.aec_w))
        if maxlv>AEC_PLIMIT:
            self.aec_w *= (AEC_PLIMIT/maxlv)
        self.aec_plimit:float = 1.2


    def start(self):
        """録音・再生を同時に開始"""
        print("start ",end="")
        self.stream = sd.Stream( samplerate=self.sample_rate,
                                blocksize=self.ds_chunk_size,
                                device = self.device, channels=1, dtype=np.int16, callback=self._callback )
        self.stream.start()
        print("----------")
        print("----------")

    def stop(self):
        """ストリームを停止"""
        print("stop ",end="")
        if self.stream:
            self.stream.stop(ignore_errors=True)

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
                if len(self.play_data)==0:
                    self.play_data.append( self.marker_pair )
                    self.play_data.append( self.zeros_pair )
                self.play_data.extend( data )
                self.is_playing = True

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
                            self.delay_factor = factor
                            self._detect_cnt=-1
                            self._detectbuf=b''
                        else:
                            print(f"[SND]delay: pos:{pos}")
                            self._detect_cnt+=1
                            self._before_pos = pos
                    else:
                        self._detect_cnt = -1
                        self.delay_samples = 0
                        self.delay_factor = 0.0
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
                        self.is_playing = False
                    play = self.zeros_pair
                self.echo_buffer.append( play.f32 )

            outdata[:,0] = play.i16[:]
        except:
            traceback.print_exc()
        finally:
            self._callback_cnt+=1

    def raw_audio(self) ->tuple[AudioF32,AudioF32]:
        with self._lock:
            x:int = math.ceil( self.delay_samples/self.ds_chunk_size )
            offset:int = self.delay_samples-x*self.ds_chunk_size
            factor:float = self.delay_factor
            e:list[AudioF32]=self.echo_buffer.copy()
            m:list[AudioI16]=self.mic_buffer.copy()
        print(f"[copy] offset {x},{offset} mic:{len(m)} spk:{len(e)}")
        mic_f32:AudioF32 = i16_to_f32( np.concatenate( m ) )
        echo_f32:AudioF32 = np.concatenate( e )
        return mic_f32,echo_f32

    def copy_audio(self) ->tuple[AudioF32,AudioF32]:
        with self._lock:
            x:int = self.delay_samples//self.ds_chunk_size
            offset:int = self.delay_samples-x*self.ds_chunk_size
            factor:float = self.delay_factor

            mics:list[AudioI16]=self.mic_buffer.copy()

            ee = len(self.echo_buffer) -x
            es = ee - len(mics)-1
            spks: list[AudioF32] = [self.zeros_f32 if l < 0 else self.echo_buffer[l] for l in range(es, ee)]

        print(f"[copy] offset {x},{offset} mic:{len(mics)} spk:{len(spks)}")
        mic_f32:AudioF32 = i16_to_f32( np.concatenate( mics ) )
        spk_f32:AudioF32 = np.concatenate( spks )
        offset = self.ds_chunk_size - offset
        spk_f32 = spk_f32[offset:offset+len(mic_f32)]
        return mic_f32,spk_f32

    def get_audio(self,keep:bool=False) ->tuple[AudioF32,AudioF32]:
        with self._lock:
            if self._detect_cnt>=0:
                return EmptyF32,EmptyF32
            x:int = self.delay_samples//self.ds_chunk_size
            offset:int = self.delay_samples-x*self.ds_chunk_size
            factor:float = self.delay_factor
        
            if keep:
                mics:list[AudioI16]=self.mic_buffer.copy()
            else:
                mics:list[AudioI16]=self.mic_buffer
                self.mic_buffer = []

            ee = len(self.echo_buffer) -x
            es = ee - len(mics)-1
            spks: list[AudioF32] = [self.zeros_f32 if l < 0 else self.echo_buffer[l] for l in range(es, ee)]

            if not keep:
                self.echo_buffer = self.echo_buffer[ee-2:]

        if len(mics)==0:
            return EmptyF32, EmptyF32
        mic_f32:AudioF32 = i16_to_f32( np.concatenate( mics ) )

        mic_max:float = np.max(np.abs(mic_f32))
        mic_lv:float = mic_max * self.mic_boost
        if mic_lv>0.9:
            self.mic_boost = (0.9/mic_max) * self.mic_boost
        mic_f32 *= self.mic_boost

        spk_f32:AudioF32 = np.concatenate( spks )
        offset = self.ds_chunk_size - offset
        spk_f32 = spk_f32[offset:offset+len(mic_f32)]
        return mic_f32,spk_f32
    
    def get_aec_audio(self) ->tuple[AudioF32,AudioF32,AudioF32]:
        mic_f32, spk_f32 = self.get_audio()
        if len(mic_f32)==0:
            return mic_f32,mic_f32,mic_f32
        lms_f32:AudioF32 = lms_echo_cancel( mic_f32, spk_f32*1.5, self.aec_mu, self.aec_w, self.aec_offset )
        return lms_f32, mic_f32, spk_f32


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

