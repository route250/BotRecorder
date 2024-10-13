import sys,os
import time
import traceback
import json

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from vosk import KaldiRecognizer, Model, SetLogLevel

sys.path.append(os.getcwd())
from BotVoice.ace_recorder import AecRecorder, AecRes, nlms_echo_cancel2, plot_aecrec
from BotVoice.rec_util import AudioF32, save_wave, load_wave, audio_info, f32_to_i16
from BotVoice.vosk_util import get_text

def run_npz(filename) ->AecRes:
    pa_chunk_size = 3200
    src:AecRes = AecRes.from_file(filename)
    sample_rate:int = src.sampling_rate
    rec:AecRecorder = AecRecorder( device=None, pa_chunk_size=pa_chunk_size, sample_rate=sample_rate)
    mic_f32 = src.raw
    spk_f32 = src.spk
    lms_f32, convergence, errors, mu = rec.nlms_echo_cancel2( mic_f32, spk_f32 )
    vadmask = convergence>0.8
    mask = vadmask.astype(np.float32)
    vad = np.zeros_like(mic_f32)
    ret:AecRes = AecRes(lms_f32, mic_f32, spk_f32, mask, vad, convergence, errors,mu)
    return ret

def run_vosk(rec:AecRes):
    pa_chunk_size = 3200
    sample_rate:int = 16000
    audio:AudioF32 = rec.audio * rec.mask
    print(f"Start recognision")
    SetLogLevel(-1)  # VOSK起動時のログ表示を抑制
    # 音声認識器を構築
    mdl:Model = Model(model_name="vosk-model-ja-0.22")
    recognizer:KaldiRecognizer = KaldiRecognizer(mdl, sample_rate)
    recognizer.SetMaxAlternatives(3)
    partialtext:str = ''
    for i in range(0,len(audio),pa_chunk_size):
        seg_f32:AudioF32 = audio[i:i+pa_chunk_size]
        i16 = f32_to_i16(seg_f32)
        if recognizer.AcceptWaveform(i16.tobytes()):
            vosk_res:dict = json.loads( recognizer.FinalResult() )
            text = get_text(vosk_res)
            print(f"[Transcrib]                         Final {text}")
            partialtext = ''
        else:
            vosk_res = json.loads( recognizer.PartialResult() )
            text = get_text(vosk_res)
            if text is not None and len(text)>0 and text!=partialtext:
                print(f"[Transcrib] Partial {text}")
                partialtext = text
    print("Done")

def main_plot_npz(filename):
    pa_chunk_size = 3200
    sample_rate:int = 16000
    npzdat:AecRes = AecRes.empty(sample_rate)
    npzdat.load(filename)
    plot_aecrec( npzdat, show=True )

def main_npz1():
    filename='test/testData/aec/out_aec_seg.npz'
    rec = run_npz(filename)
    # もしもーし、きこえますか
    run_vosk(rec)
    plot_aecrec( rec, filename='tmp/out_aec_dump.npz', show=True )

def main_npz2():
    filename='test/testData/aec/out_aec_long.npz'
    rec = run_npz(filename)
    run_vosk(rec)
    # おはようございます
    # もしもーし
    # まえだです
    # なにがすばらしいですかね
    # ざんねんでした
    # コーヒー
    # そうですかそうですか
    # ない
    plot_aecrec( rec, filename='tmp/out_aec_dump.npz', show=True )

def main_npz3():
    filename='test/testData/aec/logaudio001.npz'
    rec = run_npz(filename)
    run_vosk(rec)

def main_plot_npz3():
    filename='tmp/logdir/logaudio002.npz'
    main_plot_npz(filename)

def main_vosk():
    filename='tmp/out_aec_dump.npz'
    print(f"load {filename}")
    rec:AecRes = AecRes.from_file(filename)
    run_vosk(rec)

if __name__ == "__main__":
    #main_tome()
    #main()
    # main_npz2()
    main_plot_npz3()
    #main_vosk()
    #main_get()
    # main_file()