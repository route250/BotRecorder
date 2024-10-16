import sys,os
import traceback
from typing import NamedTuple
from queue import Queue

import numpy as np
from numpy.typing import NDArray
from scipy.fft import rfft, irfft
import sounddevice as sd
from sounddevice import CallbackFlags
from silero_vad import load_silero_vad
import torch
torch.backends.nnpack.set_flags(False)

sys.path.append(os.getcwd())
from BotVoice.rec_util import AudioI16, AudioF32, EmptyF32, np_append, save_wave, load_wave, signal_ave, sin_signal
from BotVoice.rec_util import f32_to_i16, i16_to_f32, to_f32, resample, generate_mixed_tone, audio_info
from BotVoice.rec_util import add_tone, add_white_noise

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
    convergence = np.ones(mic_len,dtype=np.float32)

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

    c_width:int = 512
    c_pos:int = 0
    c_val = 0.0
    # LMSアルゴリズムのメインループ
    for mu3 in (mu,):
        for n in range(mic_len):
                # スピーカー出力 spk の一部をスライスして使う (直近の num_taps サンプルを使う)
                spk_slice = spk_f64[n:n+num_taps]  # スライスしてタップ分の信号を取得
                # スピーカー出力がなければ処理しない
                if np.count_nonzero(spk_slice)==0:
                    cancelled_signal[n] = mic[n]
                    c_pos = 0
                    continue
                # スピーカー出力 spk_slice とフィルタ係数 w の内積によるフィルタ出力 y(n) を計算
                y = np.dot(w, spk_slice)
                # エラー e(n) を計算 (マイク信号 mic[n] とフィルタ出力 y の差)
                e = mic[n] - y
                # エコーキャンセル後の信号を計算 (マイク信号から予測されたエコーを引く)
                cancelled_signal[n] = e
                errors[n] = e
                # 収束の程度を判定
                if c_pos==0:
                    c_val, peak_index = evaluate_convergence( w )
                c_pos = (c_pos+1)%c_width
                convergence[n] = c_val
                # フィルタ係数の更新式
                factor = mu_factor[n] # np.dot(spk_slice, spk_slice)
                w[:] = w + (e*factor) * spk_slice

    return np.clip(cancelled_signal,-0.99,0.99),convergence,errors

def evaluate_convergence(coeff:NDArray[np.float64], window_factor=20) ->tuple[float,int]:
    num_taps = len(coeff)
    # widthをnum_taps/6の半分に設定
    width = int(num_taps/window_factor)
    half_width = int( width / 2)
    coeff_abs = np.abs(coeff)
    peak_index:int = int(np.argmax(coeff_abs))
    
    start_index = min(max(0, peak_index - half_width),num_taps-width)
    end_index = start_index+width
    
    peak_sum = np.sum(coeff_abs[start_index:end_index])
    total_sum = np.sum(coeff_abs)
    
    convergence = peak_sum / total_sum
    return round(convergence,2),peak_index

def buffer_update(buffer: NDArray[np.float32], seg: NDArray[np.float32]):
    # bufferがsegよりも小さいか同じ場合、segの最後の部分をbufferにコピー
    if len(buffer) <= len(seg):
        buffer[:] = seg[-len(buffer):]  # segの最後のlen(buffer)要素をコピー
    else:
        # segがbufferより小さい場合、古いデータをずらしつつsegを追加
        remaining_space = len(buffer) - len(seg)
        buffer[:remaining_space] = buffer[-remaining_space:]  # 古いデータをシフト
        buffer[remaining_space:] = seg  # segを末尾にコピー

def buffer_append(buffer: NDArray[np.float32], value: float):
    # バッファの要素を1つシフトし、最後に新しい値を追加
    buffer[:-1] = buffer[1:]
    buffer[-1] = value

def sbuffer_append(buffer: NDArray[np.float32], last_value: float) ->float:
    # 合計値を更新
    sum = buffer[0] - buffer[1] + last_value    
    # バッファの要素を1つシフトし、最後に新しい値を追加
    buffer[1:-1] = buffer[2:]
    buffer[-1] = last_value
    # 合計値を保存
    buffer[0] = sum
    return sum/(len(buffer)-1) # 平均を返す

class SpkPair(NamedTuple):
    f32:AudioF32
    i16:AudioI16

class AecRes:

    @staticmethod
    def empty(sampling_rate):
        return AecRes(EmptyF32,EmptyF32,EmptyF32,EmptyF32,EmptyF32,EmptyF32,EmptyF32,EmptyF32,sampling_rate=sampling_rate)

    @staticmethod
    def from_file(filename):
        rec = AecRes.empty(0)
        rec.load(filename)
        return rec

    def __init__(self,audio:AudioF32, raw:AudioF32, spk:AudioF32, mask:AudioF32, vad:AudioF32, convergence:AudioF32, errors:AudioF32,mu:AudioF32, *, sampling_rate:int=16000 ):
        self.sampling_rate = sampling_rate
        self.audio:AudioF32 = audio
        self.raw:AudioF32 = raw
        self.spk:AudioF32 = spk
        self.mask:AudioF32 = mask
        self.vad:AudioF32 = vad
        self.convergence:AudioF32 = convergence
        self.errors:AudioF32 = errors
        self.mu:AudioF32 = mu

    def clear(self):
        self.audio = EmptyF32
        self.raw = EmptyF32
        self.spk = EmptyF32
        self.mask = EmptyF32
        self.vad = EmptyF32
        self.convergence = EmptyF32
        self.errors = EmptyF32
        self.mu = EmptyF32

    def __len__(self):
        return len(self.audio)

    def duration(self):
        return len(self.audio)/self.sampling_rate

    def __iadd__(self, res):
        if self.audio is None or len(self.audio) == 0:
            self.audio = res.audio
            self.raw = res.raw
            self.spk = res.spk
            self.mask = res.mask
            self.vad = res.vad
            self.convergence = res.convergence
            self.errors = res.errors
            self.mu = res.mu
        else:
            self.audio = np.concatenate((self.audio, res.audio))
            self.raw = np.concatenate((self.raw, res.raw))
            self.spk = np.concatenate((self.spk, res.spk[-len(res.audio):]))
            self.mask = np.concatenate((self.mask, res.mask))
            self.vad = np.concatenate((self.vad, res.vad))
            self.convergence = np.concatenate((self.convergence, res.convergence))
            self.errors = np.concatenate((self.errors, res.errors))
            self.mu = np.concatenate((self.mu, res.mu))
        return self

    def __add__(self, res):
        new_audio = np.concatenate((self.audio, res.audio)) if self.audio is not None else res.audio
        new_raw = np.concatenate((self.raw, res.raw)) if self.raw is not None else res.raw
        new_spk = np.concatenate((self.spk, res.spk[-len(res.audio):])) if self.spk is not None else res.spk
        new_mask = np.concatenate((self.mask, res.mask)) if self.mask is not None else res.mask
        new_vad = np.concatenate((self.vad, res.vad)) if self.vad is not None else res.vad
        new_convergence = np.concatenate((self.convergence, res.convergence)) if self.convergence is not None else res.convergence
        new_errors = np.concatenate((self.errors, res.errors)) if self.errors is not None else res.errors
        new_mu = np.concatenate((self.mu, res.mu)) if self.mu is not None else res.mu

        return AecRes(new_audio, new_raw, new_spk, new_mask, new_vad, new_convergence, new_errors,new_mu, sampling_rate=self.sampling_rate)

    def save(self, filename):
        # npzでファイルに保存する
        basename = os.path.splitext(filename)[0]
        npzname = basename+".npz"
        np.savez(npzname, sampling_rate=self.sampling_rate, audio=self.audio, raw=self.raw, spk=self.spk, mask=self.mask, vad=self.vad, convergence=self.convergence, errors=self.errors,mu=self.mu)
        rawname = basename+"_mic.wav"
        save_wave(rawname,self.raw,ch=1,sampling_rate=self.sampling_rate)
        audioname = basename+"_lms.wav"
        save_wave(audioname,self.audio,ch=1,sampling_rate=self.sampling_rate)
        audioname = basename+".wav"
        save_wave(audioname,self.audio*self.mask,ch=1,sampling_rate=self.sampling_rate)

    def load(self, filename):
        # npzファイルからロードする
        try:
            data = np.load(filename)
            self.sampling_rate = int(data['sampling_rate'])
            self.audio = data['audio']
            self.raw = data['raw']
            self.spk = data['spk']
            self.mask = data['mask']
            self.vad = data['vad']
            self.convergence = data['convergence']
            self.errors = data['errors']
            self.mu = data['mu']
            if len(self.audio)!=len(self.raw) or len(self.audio)!=len(self.mask):
                print(f"Error loading file {filename} invalid array length")
        except (IOError, ValueError) as e:
            print(f"Error loading file {filename}: {e}")

class RecData(NamedTuple):
    mic:AudioF32
    spk:AudioF32

class SpkData:
    def __init__(self,data:list[SpkPair], text:str):
        self.seq: list[SpkPair] = data
        self.text: str = text

class AecRecorder:

    MarkerHz1:int=417
    MarkerHz2:int=852

    def __init__(self, device=None, pa_chunk_size:int=3200, sample_rate:int=16000):

        self.device = device
        self.ds_chunk_size = pa_chunk_size
        self.sample_rate = sample_rate
        #
        self._stream = None
        self._callback_cnt:int = 0

        # 定数
        self.zeros_f32 = np.zeros( self.ds_chunk_size, dtype=np.float32 )
        self.zeros_i16 = np.zeros( self.ds_chunk_size, dtype=np.int16 )

        # 再生データ
        self._spk_pause:bool = False
        self.spk_q:Queue[SpkData] = Queue()
        self.end_q:Queue[SpkData] = Queue()
        self.play_spk:SpkData|None = None
        self.play_list:list[SpkPair] = []
        self.play_pos:int = 0
        self._post_play_count:int = 0
        self._post_play_num:int = self.sample_rate//self.ds_chunk_size+1

        # 録音データ
        self.mic_boost:float = 1.0
        self.mic_q:Queue[RecData] = Queue()

        # 先頭マーカー検出
        self._marker_lv:float = 0.4
        tone1:AudioF32 = maek_marker_tone( self.ds_chunk_size, sample_rate, AecRecorder.MarkerHz1, AecRecorder.MarkerHz2, vol=self._marker_lv )
        self.marker_tone_f32:AudioF32 = tone1
        self.marker_tone_I16:AudioI16 = f32_to_i16( self.marker_tone_f32 )
        self.marker_bytes:bytes = f32_to_i16( tone1 ).tobytes()
        self.marker_pair:SpkPair = SpkPair( self.marker_tone_f32, self.marker_tone_I16 )
        self.zeros_pair:SpkPair = SpkPair( self.zeros_f32, self.zeros_i16 )
        self._detectbuf:AudioF32 = EmptyF32
        self._detect_num:int = int( 2 * self.sample_rate / self.ds_chunk_size )
        self._detect_cnt:int = -1
        self._before_pos:int = -1
        self.delay_samples:int = 0

        self.spk_buffer:list[AudioF32] = [self.zeros_f32] * self._detect_num

        # エコーキャンセルフィルター
        self.aec_mu = 0.2 # 学習率
        self.aec_min_mu = self.aec_mu/1000
        self.aec_max_mu = self.aec_mu*10
        self.aec_taps = 700 # フィルターの長さ
        self.aec_offset = -100 # 先頭がよくずれるので余裕を
        self.aec_convergence_pos:int = 0
        self.aec_convergence_val:float = 0.0
        self.aec_convergence_up:float = 0.25 # 音声判定レベル
        self.aec_convergence_up2:float = 0.20 # ダブルトーク判定レベル
        self.aec_convergence_up3:float = 0.20 # offset修正レベル
        self.lms_pause:int = 0
        self.lms_monitor:int = 0
        self.lms_long_buf:AudioF32 = np.zeros(self.sample_rate*2, dtype=np.float32)
        self.lms_short_buf:AudioF32 = np.zeros(int(self.sample_rate*.2), dtype=np.float32)
        # 初期係数
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
        if self._post_play_count>0 or self.spk_q.qsize()>0:
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
        print("start ",end="")
        self._stream = sd.Stream( samplerate=self.sample_rate,
                                blocksize=self.ds_chunk_size,
                                device = self.device, channels=1, dtype=np.int16, callback=self._callback )
        self._stream.start()
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

    def play(self, text:str, audio:AudioF32|None, sr:int|None=None ):
        """再生データを設定し、再生を開始"""
        # print("play ",end="")
        audio_f32:AudioF32 = to_f32(audio)
        if audio_f32 is None or len(audio_f32)==0:
            return
        if isinstance(sr,int|float) and sr>0:
            audio_f32 = resample(audio_f32,orig=sr,target=self.sample_rate)
        audio_f32 = add_white_noise( audio_f32, level=0.005 )
        max_lv = np.max(np.abs(audio_f32))
        if max_lv>0.3:
            audio_f32 *= (0.3/max_lv)
        play_f32:AudioF32 = np.concatenate( (audio_f32,self.zeros_f32) )
        play_i16:AudioI16 = f32_to_i16(play_f32)
        size:int = len(audio_f32)
        step:int = self.ds_chunk_size
        data:list[SpkPair] = [ SpkPair(play_f32[s:s+step],play_i16[s:s+step]) for s in range(0,size,step) ]
        self.spk_q.put( SpkData(data, text ) )

    def get_play_text(self) ->str:
        aa:list[SpkData] = []
        try:
            while self.end_q.qsize()>0:
                aa.append( self.end_q.get_nowait() )
        except:
            pass
        text:str = "".join( [ a.text for a in aa ])
        return text

    def cancel(self):
        """再生データを設定し、再生を開始"""
        print("[REC:canlel]",end="")
        try:
            while self.spk_q.qsize()>0:
                self.spk_q.get_nowait()
        except:
            pass

    def pause(self,b:bool):
        if self.is_playing() and self._spk_pause != b:
            if b:
                print("[REC:pause]",end="")
            else:
                print("[REC:resume]",end="")
        self._spk_pause = b

    def findfit(self, audio_f32:NDArray[np.float32], key_f32:NDArray[np.float32]) ->tuple[int,float]:
        # FFTを使った相互相関の計算とキーの位置検出
        audio_len = len(audio_f32)
        key_len = len(key_f32)
        n = audio_len + key_len - 1
        audio_fft = rfft(audio_f32, n)
        key_fft = rfft(key_f32, n)
        correlation_fft = irfft(audio_fft * np.conjugate(key_fft), n)
        j_opt = int(np.argmax(correlation_fft))
        start_idx:int = int(j_opt)
        if start_idx<key_len or audio_len-key_len <= start_idx:
            return -1, 1
        before = audio_f32[start_idx-key_len:start_idx]
        target = audio_f32[start_idx:start_idx+key_len]
        key_lv = signal_ave(np.abs(key_f32))
        before_lv = signal_ave(np.abs(before))
        target_lv = signal_ave(np.abs(target))
        if (target_lv/before_lv)<1.5:
            return -2, 1
        scale_factor = float( target_lv/key_lv ) # スケーリング係数の計算
        return start_idx, scale_factor

    def _callback(self, inbytes:np.ndarray, outdata:np.ndarray, frames:int, ctime, status: CallbackFlags ) ->None:
        if status.input_overflow:
            print("[input_overflow]")
        if status.output_underflow:
            print("[output_underflow]")
        # inbytesのサイズは Streamのblocksizeに一致する
        try:
            # もらったデータはコピーしないと後で上書きされちゃうぞ
            mic_data_f32 = i16_to_f32(inbytes[:,0])

            if 0<=self._detect_cnt:
                # 位置検出実行中
                if self._detect_cnt<self._detect_num:
                    self._detectbuf = np.concatenate( (self._detectbuf,mic_data_f32) )
                    if self._detect_cnt<5:
                        self._detect_cnt+=1
                    else:
                        pos,mfactor = self.findfit( self._detectbuf, self.marker_tone_f32 )
                        if 0.1<mfactor and mfactor <10.0 and 0<=pos and pos==self._before_pos:
                            self.mic_boost = self.mic_boost/mfactor
                            delay:int = pos + self.ds_chunk_size
                            print(f"[SND]delay: pos:{pos} factor:{mfactor} OK {delay} boost:{self.mic_boost}")
                            self.delay_samples = delay
                            self._detect_cnt=-1
                            self._detectbuf=EmptyF32
                        else:
                            print(f"[SND]delay: pos:{pos} factor:{mfactor}")
                            self._detect_cnt+=1
                            self._before_pos = pos
                else:
                    self._detect_cnt = -1
                    self.delay_samples = 0
                    self._detectbuf=EmptyF32
                    print(f"[SND]delay:NotFound")

            try:
                if self.play_spk is None and self.spk_q.qsize()>0 and not self._spk_pause:
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
                        self.end_q.put(self.play_spk)
                        self.play_spk = None
            if play is not None:
                # 再生データが設定されている場合は再生
                self._post_play_count = self._post_play_num
                if play is self.marker_pair:
                    # 位置検出データの開始
                    self._detect_cnt = 0
                    self._before_pos = -1
                    self._detectbuf = EmptyF32
                    print(f"[SND]delay:Start")
            else:
                # 再生してない
                if self._post_play_count>0:
                    self._post_play_count-=1
                play = self.zeros_pair

            outdata[:,0] = play.i16[:]
            self.mic_q.put( RecData(mic_data_f32,play.f32))
        except:
            traceback.print_exc()
        finally:
            self._callback_cnt+=1

    def get_raw_audio(self) ->tuple[AudioF32,AudioF32]:
        mic_buf:list[AudioF32] = []
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
        mic_f32:AudioF32 = np.concatenate( mic_buf ) * self.mic_boost
        spk_f32:AudioF32 = np.concatenate( spk_buf )
        spk_f32 = spk_f32[-len(mic_f32)-delay_samples-len(self.aec_w):len(spk_f32)-delay_samples]

        self.spk_buffer = [s for s in self.spk_buffer[-spk_sz:]]

        return mic_f32,spk_f32
    
    def convert_aec_audio(self,mic_f32:AudioF32,spk_f32:AudioF32) ->AecRes:
        if len(mic_f32)==0:
            return AecRes(mic_f32,mic_f32,mic_f32,mic_f32,mic_f32,mic_f32,mic_f32,mic_f32)
        lms_f32, convergence, errors,mu = self.nlms_echo_cancel2( mic_f32, spk_f32 )
        vadmask = convergence>0.7
        mask = vadmask.astype(np.float32)
        vad = np.zeros_like(lms_f32)
        ret:AecRes = AecRes(lms_f32, mic_f32, spk_f32, mask, vad, convergence, errors,mu)
        return ret

    def get_aec_audio(self) ->AecRes:
        mic_f32, spk_f32 = self.get_raw_audio()
        return self.convert_aec_audio(mic_f32,spk_f32)

    def get_vad_audio(self) ->AecRes:
        mic_f32, spk_f32 = self.get_raw_audio()
        if len(mic_f32)==0:
            return AecRes(mic_f32,mic_f32,mic_f32,mic_f32,mic_f32,mic_f32,mic_f32,mic_f32)
        lms_f32, convergence, errors, mu = self.nlms_echo_cancel2( mic_f32, spk_f32 )
        vad = self.silerovad((lms_f32))
        pre = 6400
        post = 6400+3200
        c_up = self.aec_convergence_up
        d_dn = 2.0
        vadmask = np.zeros( len(vad), dtype=np.float32 )
        for i in range(len(lms_f32)):
            v = vad[i]
            c = convergence[i]
            if v>self.vad_up and c>c_up:
                if self.vad_sw==0:
                    vadmask[ max(0,i-pre):i] = 1.0
                self.vad_sw = post
            else:
                if self.vad_sw>0:
                    self.vad_sw-=1
            vadmask[i] = 1.0 if self.vad_sw>0 else 0.0
        ret:AecRes = AecRes(lms_f32, mic_f32, spk_f32, vadmask, vad, convergence, errors, mu)
        return ret

    def get_audio(self) ->tuple[AudioF32,AudioF32]:
        ret:AecRes = self.get_aec_audio()
        return ret.audio, ret.mask

    def nlms_echo_cancel2(self, mic: AudioF32, spk_f32: AudioF32) -> tuple[AudioF32,AudioF32,AudioF32,AudioF32]:
        """
        LMSアルゴリズムによるエコーキャンセルを実行する関数。

        Parameters:
        mic (np.ndarray): マイクからの入力信号
        spk (np.ndarray): スピーカーからの出力信号

        self.mu (float): ステップサイズ（学習率）
        self.w (np.ndarray): フィルタ係数ベクトルの初期値

        Returns:
        np.ndarray: エコーキャンセル後の信号
        """
        validate_f32(mic,'mic')
        validate_f32(spk_f32, 'spk')
        validate_f64(self.aec_w,'coeff')
        mic_len = len(mic)
        num_taps = len(self.aec_w)
        if mic_len != len(spk_f32)-num_taps:
            raise TypeError("invalid array length")

        # エコーキャンセル後の信号を保存する配列
        cancelled_signal = np.zeros(mic_len,dtype=np.float32)
        # 誤差を記録する配列
        errors = np.zeros(mic_len,dtype=np.float32)
        convergence = np.ones(mic_len,dtype=np.float32)

        spk_f64 = spk_f32.astype(np.float64)

        AEC_PLIMIT:float = 1e30
        maxlv = np.sum(np.abs(self.aec_w))
        if maxlv>AEC_PLIMIT:
            print(f"[WARN] w is too large {maxlv}")
            self.aec_w *= (AEC_PLIMIT/maxlv)
        peak_index:int = -1

        # スピーカー出力の全項目の二乗を事前に計算
        spk_squared = spk_f64 ** 2
        # 音の有無
        active = np.abs(spk_f32)>0.001
        # 有効な範囲内でのみ計算を実行
        spk_on = np.zeros(mic_len,dtype=np.float64)
        factor = np.zeros(mic_len,dtype=np.float64)
        for n in range(mic_len):
            factor[n] = np.sum(spk_squared[n:n+num_taps])+1e-9
            active_rate = np.sum(active[n:n+num_taps])/num_taps
            spk_on[n] = 1.0 if active_rate>0.9 else 0.0
        mu_factor = np.clip( self.aec_mu/factor, self.aec_min_mu, self.aec_max_mu ) * spk_on
        # ダブルトーク検出用のエラーレベル
        error_rate_limit = 25
        c_width:int = 512
        # --
        lms_long_ave = self.lms_long_buf[0]/(len(self.lms_long_buf)-1)
        # LMSアルゴリズムのメインループ
        for n in range(mic_len):
            # スピーカー出力 spk の一部をスライスして使う (直近の num_taps サンプルを使う)
            spk_slice = spk_f64[n:n+num_taps]  # スライスしてタップ分の信号を取得
            # スピーカー出力がなければ処理しない
            if np.count_nonzero(spk_slice)==0:
                if self.lms_monitor!=0:
                    print(" 🔀 ",end="")
                    self.lms_monitor=0
                cancelled_signal[n] = mic[n]
                self.aec_convergence_pos = (self.aec_convergence_pos//c_width)*c_width
                continue
            # スピーカー出力 spk_slice とフィルタ係数 w の内積によるフィルタ出力 y(n) を計算
            y = np.dot(self.aec_w, spk_slice)
            # エラー e(n) を計算 (マイク信号 mic[n] とフィルタ出力 y の差)
            e = mic[n] - y
            e2 = e*e
            # エコーキャンセル後の信号を計算 (マイク信号から予測されたエコーを引く)
            cancelled_signal[n] = e
            lms_short1_ave = sbuffer_append(self.lms_short_buf,e2)
            lms_short2_ave = np.mean(self.lms_short_buf[-100:])
            lms_short_ave = max(lms_short1_ave,lms_short2_ave)
            err_rate = min( error_rate_limit*2, lms_short_ave / (lms_long_ave+1e-9) )
            errors[n] = err_rate/error_rate_limit/2
            if self.aec_convergence_val>self.aec_convergence_up2 and err_rate>error_rate_limit:
                if self.lms_pause<=0:
                    print(f"⬆️{peak_index}",end="")
                self.lms_pause = self.ds_chunk_size
            else:
                if self.lms_pause>0:
                    self.lms_pause -= 1
                    if self.lms_pause<=0:
                        print(f"⬇️",end="")
            if self.aec_convergence_val<self.aec_convergence_up2 or self.lms_pause<=0:
                lms_long_ave = sbuffer_append(self.lms_long_buf,e2)
                # フィルタ係数の更新式
                factor = mu_factor[n] # np.dot(spk_slice, spk_slice)
                self.aec_w[:] = self.aec_w + (e*factor) * spk_slice
                # 収束の程度を判定
                if (self.aec_convergence_pos%c_width)==0:
                    self.aec_convergence_val, peak_index = evaluate_convergence( self.aec_w )
                self.aec_convergence_pos += 1
                convergence[n] = self.aec_convergence_val
                if self.lms_monitor!=1:
                    print(f" ⤴️ ",end="")
                    self.lms_monitor=1
            else:
                if self.aec_convergence_val>self.aec_convergence_up:
                    convergence[n] = 0.9
                else:
                    convergence[n] = 0.0
                if self.lms_monitor!=2:
                    print(f" ⤵︎ ",end="")
                    self.lms_monitor=2

        if peak_index>0 and self.aec_convergence_val>self.aec_convergence_up3:
            # 位置オフセットを補正する
            baseidx = len(self.aec_w)-20
            diff = baseidx-peak_index
            if diff != 0:
                print(f"[p:{peak_index},{diff},{self.aec_offset}]",end="")
                self.shift_value_and_clear(diff)

        return np.clip(cancelled_signal,-0.99,0.99),convergence,errors,mu_factor.astype(np.float32)

    def shift_value_and_clear(self, shift_amount):
        # 移動量を計算
        self.aec_offset += shift_amount        
        # np.rollでシフト
        shifted_arr = np.roll(self.aec_w, shift_amount)
        # シフトの方向に応じて、回り込む部分をゼロクリア
        if shift_amount > 0:
            # 右にシフトした場合、最初の shift_amount 個をゼロクリア
            shifted_arr[:shift_amount] = 0
        elif shift_amount < 0:
            # 左にシフトした場合、最後の abs(shift_amount) 個をゼロクリア
            shifted_arr[shift_amount:] = 0
        self.aec_w[:] = shifted_arr
        self.aec_convergence_val = 0
        self.aec_convergence_pos = 1

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

def plot_aecrec( rec:AecRes, *, filename:str|None=None, show:bool=False ):
    basename:str|None = os.path.splitext(filename)[0] if filename is not None else None
    if basename is not None:
        print(f"[OUT] save {basename} {audio_info(rec.audio,sample_rate=rec.sampling_rate)}")
        rec.save( filename )

    max_y = round( 0.05 + max( np.max(np.abs(rec.raw)), np.max(np.abs(rec.spk)), np.max(np.abs(rec.audio)) ), 1 )
    x1 = np.array( range(len(rec.raw)) )
    x2 = np.array( range(len(rec.spk))) - (len(rec.raw)-len(rec.spk))
    mask_bool:list[bool] = [ m>0.0 for m in rec.mask ]

    import matplotlib.pyplot as plt

    plt.figure(figsize=(12,3))
    plt.plot(x1,rec.raw, label='Mic')
    plt.plot(x2,rec.spk, label='Spk', alpha=0.2)
    plt.ylim(-max_y,max_y)
    plt.legend()
    if basename is not None:
        plt.savefig(f'{basename}_spk.png',dpi=300)

    # 図の作成
    fig, ax1 = plt.subplots(figsize=(12, 3))
    # Y1軸 (Mic と Lms Mic) にデータをプロット
    ax1.fill_between(x1, -max_y, max_y, where=mask_bool, color='green', lw=0, alpha=0.1, label='mask' ) 
    ax1.plot(x1,rec.raw, label='Mic', alpha=0.3)
    ax1.plot(x1,rec.audio, label='LMS', alpha=0.3)
    ax1.set_ylim(-max_y,max_y)
    ax1.set_ybound(-1.0,1.0)
    ax1.set_ylabel('signal')
    ax1.legend(loc='upper left')
    # Y2軸 Maskをプロット
    ax2 = ax1.twinx()    # 画像を保存
    ax2.plot(x1,rec.vad, label='vad', color='orange', alpha=0.1)
    ax2.plot(x1,rec.convergence, label='convergence', color='yellow', alpha=0.9)
    ax2.plot(x1,rec.errors, label='eror', color='red', alpha=0.3)
    ax2.set_ybound(0.0,1.0)
    ax2.set_ylabel('rate')
    ax2.legend(loc='upper right')
    if basename is not None:
        plt.savefig(f'{basename}_lms.png', dpi=300)

    # 図の作成
    fig, ax1 = plt.subplots(figsize=(12, 3))
    # Y1軸 Errorsをプロット
    ax1.plot(x1,rec.errors, color='red', label='Errors')
    ax1.set_ylabel('Errors')
    ax1.legend(loc='upper left')
    # Y2軸 Maskをプロット
    ax2 = ax1.twinx()
    ax2.fill_between(x1, 0, rec.mu, color='green', lw=0, alpha=0.8, label='mu' ) 
    #ax2.set_ybound(0.0,1.0)
    ax2.set_ylabel('mu')
    ax2.legend(loc='upper right')
    if basename is not None:
        plt.savefig(f'{basename}_errors.png', dpi=300)

    # グラフを表示
    if basename is None or show:
        plt.show()
