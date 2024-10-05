import sys,os
import time
import traceback
from typing import NamedTuple
from queue import Queue
import math
import audioop

import numpy as np
from numpy.typing import NDArray
import sounddevice as sd
from sounddevice import CallbackFlags
from silero_vad import load_silero_vad
import torch
torch.backends.nnpack.set_flags(False)

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

def nlms_echo_cancel(mic: AudioF32, spk_f32: AudioF32, mu: float, w: NDArray[np.float64]) -> tuple[float,AudioF32,NDArray[np.float32]]:
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
    # 誤差を記録する配列
    errors = np.zeros(mic_len,dtype=np.float32)

    spk_f64 = spk_f32.astype(np.float64)

    AEC_PLIMIT:float = 1e30
    maxlv = np.sum(np.abs(w))
    if maxlv>AEC_PLIMIT:
        print(f"[WARN] w is too large {maxlv}")
        w *= (AEC_PLIMIT/maxlv)

    # スピーカー出力の全項目の二乗を事前に計算
    spk_squared = spk_f64 ** 2
    # 音の有無
    active = np.abs(spk_f32)>0.0001
    # 有効な範囲内でのみ計算を実行
    r = np.zeros(mic_len,dtype=np.float64)
    spk_on = np.zeros(mic_len,dtype=np.float64)
    factor = np.zeros(mic_len,dtype=np.float64)
    for n in range(mic_len):
        factor[n] = np.sum(spk_squared[n:n+num_taps])+1e-9
        r[n] = np.sum(active[n:n+num_taps])/num_taps
        spk_on[n] = 1.0 if r[n]>0.9 else 0.0
    mu_factor = np.clip( mu/factor, mu/100, mu ) * spk_on
    spk_rate:float = float(np.sum(spk_on)/len(spk_on)) 
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
                errors[n] = e
                # フィルタ係数の更新式
                factor = mu_factor[n] # np.dot(spk_slice, spk_slice)
                # if norm_factor != np.dot(spk_slice, spk_slice):
                #     print(f"ERROR:invalid norm_factor")
                if factor >1e-3:
                    w[:] = w + (e*factor) * spk_slice
                    # if np.isnan(w).any() or np.isinf(w).any():
                    #     raise ValueError("spk include NaN or INF")

    return spk_rate, np.clip(cancelled_signal,-0.99,0.99),errors

def validate_f32(arr:np.ndarray, name ):
    try:
        if len(arr.shape) != 1:
            raise TypeError(f"{name} must be a 1-dimensional array")
        if arr.dtype != np.float32:
            raise TypeError(f"{name} must be of dtype np.float32")
        if np.isnan(arr).any() or np.isinf(arr).any():
            raise ValueError(f"{name} includes NaN or INF")
    except:
        raise TypeError(f"{name} must be a numpy array")

def validate_f64(arr:np.ndarray, name ):
    try:
        if len(arr.shape) != 1:
            raise TypeError(f"{name} must be a 1-dimensional array")
        if arr.dtype != np.float64:
            raise TypeError(f"{name} must be of dtype np.float64")
        if np.isnan(arr).any() or np.isinf(arr).any():
            raise ValueError(f"{name} includes NaN or INF")
    except:
        raise TypeError(f"{name} must be a numpy array")

def nlms_echo_cancel2(mic: AudioF32, spk_f32: AudioF32, mu: float, w: NDArray[np.float64]) -> tuple[AudioF32,AudioF32,AudioF32]:
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
    validate_f32(mic,'mic')
    validate_f32(spk_f32, 'spk')
    validate_f64(w,'coeff')
    mic_len = len(mic)
    num_taps = len(w)
    if mic_len != len(spk_f32)-num_taps:
        raise TypeError("invalid array length")

    # エコーキャンセル後の信号を保存する配列
    cancelled_signal = np.zeros(mic_len,dtype=np.float32)
    # 誤差を記録する配列
    errors = np.zeros(mic_len,dtype=np.float32)
    mask = np.ones(mic_len,dtype=np.float32)

    spk_f64 = spk_f32.astype(np.float64)

    AEC_PLIMIT:float = 1e30
    maxlv = np.sum(np.abs(w))
    if maxlv>AEC_PLIMIT:
        print(f"[WARN] w is too large {maxlv}")
        w *= (AEC_PLIMIT/maxlv)

    # スピーカー出力の全項目の二乗を事前に計算
    spk_squared = spk_f64 ** 2
    # 音の有無
    active = np.abs(spk_f32)>0.0001
    # 有効な範囲内でのみ計算を実行
    r = np.zeros(mic_len,dtype=np.float64)
    spk_on = np.zeros(mic_len,dtype=np.float64)
    factor = np.zeros(mic_len,dtype=np.float64)
    for n in range(mic_len):
        factor[n] = np.sum(spk_squared[n:n+num_taps])+1e-9
        r[n] = np.sum(active[n:n+num_taps])/num_taps
        spk_on[n] = 1.0 if r[n]>0.9 else 0.0
    mu_factor = np.clip( mu/factor, mu/100, mu ) * spk_on

    concentration:float = 0.0
    cw:int = 500
    cx:int = 0
    c_hi:float = 3.5
    c_lo:float = 3.0
    c_val = 0.0
    # LMSアルゴリズムのメインループ
    for mu3 in (mu,):
        for n in range(mic_len):
                # スピーカー出力 spk の一部をスライスして使う (直近の num_taps サンプルを使う)
                spk_slice = spk_f64[n:n+num_taps]  # スライスしてタップ分の信号を取得
                # スピーカー出力がなければ処理しない
                if np.count_nonzero(spk_slice)==0:
                    cancelled_signal[n] = mic[n]
                    c_val = 1.0
                    continue
                # スピーカー出力 spk_slice とフィルタ係数 w の内積によるフィルタ出力 y(n) を計算
                y = np.dot(w, spk_slice)
                # エラー e(n) を計算 (マイク信号 mic[n] とフィルタ出力 y の差)
                e = mic[n] - y
                # エコーキャンセル後の信号を計算 (マイク信号から予測されたエコーを引く)
                cancelled_signal[n] = e
                errors[n] = e
                # 収束の程度を判定
                if cx==0:
                    concentration = evaluate_concentration( w )
                cx = (cx+1)%cw
                if concentration<c_lo:
                    c_val = 0.0
                elif concentration>c_hi:
                    c_val = 1.0
                mask[n] = c_val
                # フィルタ係数の更新式
                factor = mu_factor[n] # np.dot(spk_slice, spk_slice)
                w[:] = w + (e*factor) * spk_slice

    return np.clip(cancelled_signal,-0.99,0.99),mask,errors

def evaluate_concentration(coeff:NDArray[np.float64], window_factor=6) ->float:
    num_taps = len(coeff)
    # widthをnum_taps/6の半分に設定
    half_width = int(num_taps / window_factor / 2)
    coeff_abs = np.abs(coeff)
    max_index_abs = np.argmax(coeff_abs)
    
    start_index = max(0, max_index_abs - half_width)
    end_index = min(max_index_abs + half_width, num_taps)
    window_size = end_index - start_index
    scaling_factor = num_taps / window_size
    
    local_sum = np.sum(coeff_abs[start_index:end_index])
    scaled_total_sum = np.sum(coeff_abs) / scaling_factor
    
    rate = local_sum / scaled_total_sum
    return rate

class SpkPair(NamedTuple):
    f32:AudioF32
    i16:AudioI16

class AecRes(NamedTuple):
    audio:AudioF32
    raw:AudioF32
    spk:AudioF32
    mask:AudioF32
    vad:AudioF32
    errors:AudioF32

class RecData(NamedTuple):
    mic:AudioI16
    spk:AudioF32

class SpkData:
    def __init__(self,data:list[SpkPair], text:str):
        self.seq: list[SpkPair] = data
        self.text: str = text

class AecRecorder:

    H1:int=440
    H2:int=880

    def __init__(self, device=None, pa_chunk_size:int=3200, sample_rate:int=16000):

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
        self.spk_q:Queue[SpkData] = Queue()
        self.play_spk:SpkData|None = None
        self.play_list:list[SpkPair] = []
        self.play_pos:int = 0
        self._is_playing = False
        self._post_play_count:int = 0
        self._post_play_num:int = 0

        # 録音データ
        self.mic_q:Queue[RecData] = Queue()
        self.mic_boost:float = 1.0
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

        # VAD
        self.vad_model = load_silero_vad()
        self.vad_sw:float = 0.0
        self.vad_up:float = 0.6
        self.vad_dn:float = 0.4

    def get_aec_coeff(self) ->NDArray[np.float64]:
        return self.aec_w.copy()

    def save_aec_coeff(self,filepath):
        np.savez_compressed(filepath,aec_coeff=self.get_aec_coeff())

    def set_aec_coeff(self,coeff:NDArray[np.float64]):
        if len(coeff) == len(self.aec_w):
            self.aec_w[:] = coeff.copy().astype(np.float64)

    def load_aec_coeff(self,filepath):
        zip = np.load(filepath)
        coeff = zip.get('aec_coeff',None)
        if coeff is not None:
            self.set_aec_coeff(coeff)

    def is_playing(self) ->int:
        if self._is_playing:
            return 1
        return 0

    def is_active(self) ->bool:
        if self._stream:
            return self._stream.active
        else:
            print("not open?")
            return False
    
    def is_stopped(self):
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
        data:list[SpkPair] = [ self.zeros_pair, self.marker_pair, self.zeros_pair ]
        self.spk_q.put( SpkData(data,'') )
        self._is_playing = True

    def play(self, text:str, audio:AudioF32|None, sr:int|None=None ):
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
            self.spk_q.put( SpkData(data, text ) )
            self._is_playing = True

    def cancel(self):
        """再生データを設定し、再生を開始"""
        print("canlel ",end="")
        try:
            while self.spk_q.qsize()>0:
                self.spk_q.get_nowait()
        except:
            pass

    def _callback(self, inbytes:np.ndarray, outdata:np.ndarray, frames:int, ctime, status: CallbackFlags ) ->None:
        if status.input_overflow:
            print("[input_overflow]")
        if status.output_underflow:
            print("[output_underflow]")
        # inbytesのサイズは Streamのblocksizeに一致する
        st:float = time.time()
        try:
            # もらったデータはコピーしないと後で上書きされちゃう
            mic_data:AudioI16 = inbytes[:,0].copy()

            if 0<=self._detect_cnt:
                # 位置検出実行中
                if self._detect_cnt<self._detect_num:
                    self._detectbuf += mic_data.tobytes()
                    pos,factor = audioop.findfit( self._detectbuf, self.marker_bytes ) if self._detect_cnt>5 else (0,0.0)
                    if self._detect_cnt>5 and 0<=pos and pos==self._before_pos:
                        tmp = self._detectbuf[pos*2:pos*2+self.ds_chunk_size]
                        lo,hi = audioop.minmax( tmp, 2 )
                        maxlv = (abs(lo)+abs(hi))/2/32768
                        self.mic_boost = self._marker_lv/maxlv if maxlv>0 else 1
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

            try:
                if self.play_spk is None and self.spk_q.qsize()>0:
                    self.play_spk = self.spk_q.get_nowait()
                    self.play_list = self.play_spk.seq
                    self.play_pos = 0
            except:
                print("!",end="")
                pass
            play:SpkPair|None = None
            if self.play_spk is not None:
                if self.play_pos<len(self.play_list):
                    play = self.play_list[self.play_pos]
                    self.play_pos += 1
                    if self.play_pos>=len(self.play_list):
                        self.play_spk = None
            if play is not None:
                # 再生データが設定されている場合は再生
                self._post_play_count = self._post_play_num
                if play is self.marker_pair:
                    # 位置検出データの開始
                    self._detect_cnt = 0
                    self._before_pos = -1
                    self._detectbuf = b''
                    print(f"[SND]delay:Start")
            else:
                # 再生してない
                if self._post_play_count>0:
                    self._post_play_count-=1
                else:
                    self._is_playing = False
                play = self.zeros_pair

            outdata[:,0] = play.i16[:]
            self.mic_q.put( RecData(mic_data,play.f32))
        except:
            traceback.print_exc()
        finally:
            et:float = time.time()
            tt = et-st
            if tt>0.01:
                print(f"[{tt:.3f}]", end="")
            self._callback_cnt+=1

    def get_raw_audio(self) ->tuple[AudioF32,AudioF32]:
        mic_buf:list[AudioI16] = []
        spk_sz:int = len(self.spk_buffer)
        try:
            rec:RecData = self.mic_q.get(timeout=0.2)
            mic_buf.append(rec.mic)
            self.spk_buffer.append(rec.spk)
            while True:
                rec:RecData = self.mic_q.get_nowait()
                mic_buf.append(rec.mic)
                self.spk_buffer.append(rec.spk)
        except:
            pass

        delay_samples = max(0,self.delay_samples+self.aec_offset)
        x:int = int( (delay_samples+len(self.aec_w))/self.ds_chunk_size ) + 1
        spk_buf: list[AudioF32] = self.spk_buffer[-len(mic_buf)-x:]

        if len(mic_buf)==0:
            return EmptyF32, EmptyF32
        mic_f32:AudioF32 = i16_to_f32( np.concatenate( mic_buf ) ) * self.mic_boost
        spk_f32:AudioF32 = np.concatenate( spk_buf )
        spk_f32 = spk_f32[-len(mic_f32)-delay_samples-len(self.aec_w):len(spk_f32)-delay_samples]

        self.spk_buffer = [s for s in self.spk_buffer[-spk_sz:]]

        return mic_f32,spk_f32

    # def get_raw_audiopppp(self,keep:bool=False) ->tuple[AudioF32,AudioF32]:
    #     with self._lock:
    #         if self._detect_cnt>=0:
    #             return EmptyF32,EmptyF32
    #         if keep:
    #             mic_buf:list[AudioI16]=self.mic_buffer.copy()
    #         else:
    #             mic_buf:list[AudioI16]=self.mic_buffer
    #             self.mic_buffer = []

    #         delay_samples = max(0,self.delay_samples+self.aec_offset)
    #         x:int = int( (delay_samples+len(self.aec_w))/self.ds_chunk_size ) + 1
    #         spk_buf: list[AudioF32] = self.spk_buffer[-len(mic_buf)-x:].copy()
    #         if not keep:
    #             self.spk_buffer = self.spk_buffer[-self._detect_num:]

    #     if len(mic_buf)==0:
    #         return EmptyF32, EmptyF32
    #     mic_f32:AudioF32 = i16_to_f32( np.concatenate( mic_buf ) ) * self.mic_boost
    #     spk_f32:AudioF32 = np.concatenate( spk_buf )
    #     spk_f32 = spk_f32[-len(mic_f32)-delay_samples-len(self.aec_w):len(spk_f32)-delay_samples]
    #     return mic_f32,spk_f32
    
    def get_aec_audio(self) ->AecRes:
        mic_f32, spk_f32 = self.get_raw_audio()
        if len(mic_f32)==0:
            return AecRes(mic_f32,mic_f32,mic_f32,mic_f32,mic_f32,mic_f32)
        lms_f32, mask, errors = nlms_echo_cancel2( mic_f32, spk_f32, self.aec_mu, self.aec_w )
        # lms_f32:AudioF32 = rls_echo_cancel( mic_f32, spk_f32, 0.98, 100, self.aec_w, self.aec_offset )
        vad = self.silerovad((lms_f32))
        for i,v in enumerate(vad):
            if self.vad_sw==0.0:
                if v>self.vad_up:
                    self.vad_sw = 1.0
            else:
                if v<self.vad_dn:
                    self.vad_sw = 0.0
            mask[i] *= self.vad_sw
        ret:AecRes = AecRes(lms_f32, mic_f32, spk_f32, mask, vad, errors)
        return ret

    def get_audio(self) ->tuple[AudioF32,AudioF32]:
        ret:AecRes = self.get_aec_audio()
        return ret.audio, ret.mask

    def silerovad( self, x:AudioF32 ) ->AudioF32:
        chunk_size = 512
        sr=16000
        # 処理する範囲を最初に計算 (512の倍数に丸める)
        l = (len(x) // chunk_size) * chunk_size
        ret:NDArray[np.float32] = np.zeros_like(x)
        t:torch.Tensor = torch.Tensor(x)
        for i in range(0,l,chunk_size):
            chunk = t[i:i+chunk_size]
            prob = self.vad_model(chunk,sr)
            ret[i:i+chunk_size] = float(prob[0][0])
        return ret

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

