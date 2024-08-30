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
from BotVoice.rec_util import f32_to_i8
from BotVoice.bot_voice import BotVoice,RATE,CHUNK_SEC,CHUNK_LEN

def main():

    model_size = "large-v3"
    model_size = "tiny" # 39M
    model_size = "base" # 74M
    model_size = "small" # 244M
    model_size = "medium" # 769M
    model_size = "large" # 1550M
    # Run on GPU with FP16
    #model = WhisperModel(model_size, device="gpu", compute_type="float16")
    # or run on GPU with INT8
    # model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
    # or run on CPU with INT8
    # model = WhisperModel(model_size, device="cpu", compute_type="int8")
    model = WhisperModel(model_size, device="auto", cpu_threads=8, compute_type="int8")

    bot_voice:BotVoice = BotVoice()

    threshold:int = 10
    audio_i8:AudioI8 = np.zeros( RATE*3, dtype=np.int8 )
    print("チェック")
    model.transcribe(audio_i8, language='ja', beam_size=5)

    text_list:list[str] = []
    delta_sec:float = 0.6
    # 指定した期間録音
    print("録音と再生を開始します...")
    start_time:float = time.time()
    stop_time:float = start_time + 180.0
    lll=0.0
    bot_voice.start()
    while bot_voice.is_active():
        now:float = time.time()
        if stop_time<=now:
            print("stop by time")
            break
        raw_seg_f32 = bot_voice.get_audio()
        if len(raw_seg_f32)<=0:
            time.sleep(0.2)
            continue
        lll=time.time()
        a_i8:AudioI8 = f32_to_i8(raw_seg_f32)
        audio_i8 = np.concatenate( (audio_i8,a_i8) )
        audio_sec:float = round( len(audio_i8)/RATE, 2 )
        audio_max:int = audio_i8.max()
        seg_list:list[Segment] = []
        if audio_max>threshold:
            segments, info = model.transcribe(audio_i8, language='ja', beam_size=5)
            for seg in segments:
                if 'ご視聴ありがとう' not in seg.text:
                    seg_list.append(seg)

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
        print(f"#Audio len:{audio_sec:.3f}s,{len(audio_i8)} split:{split_sec:.3f}s,{split_idx}")
        print()
        if split_idx>0:
            audio_i8 = audio_i8[split_idx:]
        #time.sleep( .2 )
    print("終了しました")
    bot_voice.stop()


if __name__ == "__main__":
    #mlstest()
    main()