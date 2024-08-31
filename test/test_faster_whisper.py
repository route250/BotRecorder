import sys,os
import time
import json
import wave
from io import BytesIO
from typing import Mapping
import numpy as np
import pyaudio

import matplotlib.pyplot as plt
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment
#help(WhisperModel)

sys.path.append(os.getcwd())

from BotVoice.rec_util import AudioI8, AudioI16, AudioF32, EmptyF32, np_append, save_wave, load_wave, signal_ave, sin_signal
from BotVoice.rec_util import from_f32
from BotVoice.bot_voice import BotVoice,RATE,CHUNK_SEC,CHUNK_LEN

def main():

    model_size = "tiny" # 39M
    model_size = "base" # 74M
    # model_size = "small" # 244M
    # model_size = "medium" # 769M
    # model_size = "large" # 1550M
    # model_size = "large-v3"

    # Run on GPU with FP16
    # model = WhisperModel(model_size, device="gpu", compute_type="float16")
    # or run on GPU with INT8
    # model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
    # or run on CPU with INT8
    # model = WhisperModel(model_size, device="cpu", compute_type="int8")

    Dev = "cpu"
    Typ = np.int8
    if Typ == np.float32:
        ctyp="float32"
        threshold = 0.2
    elif Typ == np.float16:
        ctyp="float16"
        threshold = 0.2
    elif Typ == np.int16:
        ctyp="int16"
        threshold = int(32767)*0.2
    elif Typ == np.int8:
        ctyp="int8"
        threshold = int(127)*0.2
    else:
        raise ValueError(f"invalid data type {Typ}")

    print(f"Load model {model_size} {Dev} {ctyp}")
    model = WhisperModel(model_size, device=Dev, compute_type=ctyp)
    print(f"load done")

    audio_np:np.ndarray = np.zeros( RATE*3, dtype=Typ )
    print("Check model")
    model.transcribe(audio_np, language='ja', beam_size=5)

    text_list:list[str] = []
    delta_sec:float = 0.6
    # 指定した期間録音
    print("録音と再生を開始します...")
    start_time:float = time.time()
    stop_time:float = start_time + 180.0
    lll=0.0
    bot_voice:BotVoice = BotVoice()
    bot_voice.start()
    while bot_voice.is_active():
        now:float = time.time()
        if stop_time<=now:
            print("stop by time")
            break
        seg_f32 = bot_voice.get_audio()
        if len(seg_f32)<=0:
            time.sleep(0.2)
            continue

        seg_np:np.ndarray = from_f32(seg_f32,dtype=Typ)
        audio_np = np.concatenate( (audio_np,seg_np) )
        audio_sec:float = round( len(audio_np)/RATE, 2 )
        audio_max:int = audio_np.max()

        print()
        transcribe_time:float = 0.0
        seg_list:list[Segment] = []
        if audio_max>threshold:
            st = time.time()
            segments, info = model.transcribe(audio_np, language='ja', beam_size=5)
            for seg in segments:
                if 'ご視聴ありがとう' not in seg.text:
                    seg_list.append(seg)
            transcribe_time = time.time()-st

        split_sec:float
        if len(seg_list)>0:
            split_sec = max(0,seg_list[0].start - delta_sec)
            # どこまで確定したか？
            fixed_len:int = 0
            for idx,seg in enumerate(seg_list):
                next_sec:float = seg_list[idx+1].start if idx+1<len(seg_list) else audio_sec
                if seg.end<audio_sec or (next_sec - seg.end)>=delta_sec:
                    fixed_len=idx+1
                    split_sec=seg.end

            print(f"Segs {fixed_len}/{len(seg_list)} {split_sec:.3f}s/{audio_sec:.3f}s Detected {info.language} {info.language_probability:.1f}%")
            for idx,seg in enumerate(seg_list):
                if idx<fixed_len:
                    # 確定
                    print("   [%.2fs-%.2fs] %s" % (seg.start, seg.end, seg.text))
                    text_list.append(seg.text)
                else:
                    print("    [%.2fs-%.2fs] ### %s" % (seg.start, seg.end, seg.text))
        else:
            split_sec = max(0,audio_sec - delta_sec)

        split_idx = int( split_sec*RATE )
        print(f"#Audio {transcribe_time:.3f}s len:{audio_sec:.3f}s,{len(audio_np)} split:{split_sec:.3f}s,{split_idx}")
        if split_idx>0:
            audio_np = audio_np[split_idx:]
        #time.sleep( .2 )
    print("終了しました")
    bot_voice.stop()


if __name__ == "__main__":
    #mlstest()
    main()