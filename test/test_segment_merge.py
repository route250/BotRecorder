import sys,os
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())
from BotVoice.segments import merge_segment, Segment, Word

def dump(segments):
    for s in segments:
        print(f"[{s.start:.3f}-{s.end:.3f}]{s.text}")

def test1():
    seg_list:list[Segment] = []
    tmdata:list[list[float]] = [
        [0.1, 0.1, 0.2 ],
        [0.1, 0.1, 0.1 ],
        [0.1, 0.1, 0.1 ],
    ]
    tm:float=0.0
    for iseg,st in enumerate(tmdata):
        seg_start = tm
        words:list[Word] = []
        for iword,w in enumerate(st):
            words.append( Word(
                start=tm,
                end=tm+w,
                word=f"w{iseg}-{iword}",
                probability=0.5
            ) )
            tm+=w
        seg_text = ','.join( [w.word for w in words])
        seg_end=tm
        seg_list.append( Segment(
            id=iseg,
            seek=iseg,
            start=seg_start,
            end=seg_end,
            text=seg_text,
            tokens=[],
            temperature=1.0,
            avg_logprob=0.5,
            compression_ratio=0.5,
            no_speech_prob=0.5,
            words=words
        ))
    print("--INPUT------")
    dump(seg_list)
    print("-------------")

    res:list[Segment] = merge_segment( seg_list, gap=0.5 )
    print("--OUTPUT------")
    dump(res)
    print("-------------")

if __name__ == "__main__":
    test1()