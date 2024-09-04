import sys,os
import time
import json
import wave
from io import BytesIO
from typing import Iterable
import numpy as np
import pyaudio

import matplotlib.pyplot as plt
import torch
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment as faster_segment, TranscriptionInfo

#help(WhisperModel)

sys.path.append(os.getcwd())

from BotVoice.rec_util import AudioI8, AudioI16, AudioF32, EmptyF32, np_append, save_wave, load_wave, signal_ave, sin_signal
from BotVoice.rec_util import from_f32
from BotVoice.segments import is_accept
from BotVoice.segments import TranscribRes, Segment, Word
from BotVoice.bot_audio import BotAudio,RATE,CHUNK_SEC,CHUNK_LEN
from BotVoice.voice_base import VoiceBase


            # {
            #   "text": "ご視聴ありがとうございました",
            #   "segments": [
            #      {
            #        "id": 0,
            #        "seek": 0,
            #        "start": 2.4,
            #        "end": 4.18,
            #        "text": "ご視聴ありがとうございました",
            #        "tokens": [50364, 9991, 27333, 8171, 112, 38538, 50584],
            #        "temperature": 0.0,
            #        "avg_logprob": -0.9936987161636353,
            #        "compression_ratio": 0.8571428571428571,
            #        "no_speech_prob": 0.7222453951835632,
            #        "words": [
            #          {"word": "ご", "start": 2.4, "end": 3.36, "probability": 0.0025025999639183283},
            #          {"word": "視", "start": 3.36, "end": 3.84, "probability": 0.19475798308849335},
            #          {"word": "聴", "start": 3.84, "end": 3.84, "probability": 0.9998210072517395},
            #          {"word": "ありがとうございました", "start": 3.84, "end": 4.18, "probability": 0.6102240681648254}
            #         ]
            #      }
            #    ],
            #   "language": "ja"}'



class FasterVoice(VoiceBase):

    # Run on GPU with FP16
    # model = WhisperModel(model_size, device="gpu", compute_type="float16")
    # or run on GPU with INT8
    # model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
    # or run on CPU with INT8
    # model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def __init__(self, model:str):
        self.model_size = model

    def load_model(self):

        if torch.cuda.is_available():
            Dev='cuda'
            compute_type='int8_float16'
        # elif torch.backends.mps.is_available():
        #     Dev=torch.device('mps')
        else:
            Dev='cpu'
            compute_type='int8'

        print(f"Load model {self.model_size} {Dev}")
        self.model = WhisperModel(self.model_size, device=Dev, compute_type=compute_type )

    def check_model(self):
        check_audio:AudioF32 = sin_signal()
        res:TranscribRes = self.transcrib(check_audio)

    def transcrib(self, audio_np:AudioF32) ->TranscribRes:
        iter,info = self.model.transcribe(audio_np, language='ja', beam_size=2, word_timestamps=True, temperature=0.0 )
        seg_list:list = []
        for seg in iter:
            d = seg._asdict()
            if seg.words:
                d['words'] = [ w._asdict() for w in seg.words ]
            seg_list.append(d)
        text:str = ''.join( [seg['text'] for seg in seg_list])
        data:dict = {
            'text': text,
            'segments': seg_list,
            'language': info.language
        }
        ret:TranscribRes = TranscribRes(data)
        return ret


def main():

    # Run on GPU with FP16
    # model = WhisperModel(model_size, device="gpu", compute_type="float16")
    # or run on GPU with INT8
    # model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
    # or run on CPU with INT8
    # model = WhisperModel(model_size, device="cpu", compute_type="int8")

    if torch.cuda.is_available():
        Dev='cuda'
        compute_type='int8_float16'
    # elif torch.backends.mps.is_available():
    #     Dev=torch.device('mps')
    else:
        Dev='cpu'
        compute_type='int8'

    model_size = "tiny" # 39M
    model_size = "base" # 74M
    # model_size = "small" # 461M
    # model_size = "medium" # 769M
    # model_size = "large" # 1550M
    # model_size = "large-v3"

    print(f"Load model {model_size} {Dev}")
    model = WhisperModel(model_size, device=Dev, compute_type=compute_type )
    print(f"load done")

    audio_np:np.ndarray = np.zeros( RATE*3, dtype=np.float32 )
    print("Check model")
    segments, info = model.transcribe(audio_np, language='ja', beam_size=2, word_timestamps=True )

    audio_np:np.ndarray = np.zeros( RATE*3, dtype=np.float32 )
    text_list:list[str] = []
    delta_sec:float = 0.6
    threshold:float = 0.01
    # 指定した期間録音
    print("録音と再生を開始します...")
    start_time:float = time.time()
    stop_time:float = start_time + 180.0

    bot_voice:BotAudio = BotAudio()
    bot_voice.start()
    pre_segs:list[str] = []
    while bot_voice.is_active():
        now:float = time.time()
        if stop_time<=now:
            print("stop by time")
            break
        seg_f32 = bot_voice.get_audio()
        if len(seg_f32)<=0:
            time.sleep(0.2)
            continue

        audio_np = np.concatenate( (audio_np,seg_f32) )
        audio_sec:float = round( len(audio_np)/RATE, 2 )
        audio_max:int = audio_np.max()

        print()
        transcribe_time:float = 0.0
        text = ''
        split_sec = audio_sec - delta_sec
        if audio_max>threshold:
            prompt = ' '.join( text_list[-2:])
            st = time.time()
            segments, info = model.transcribe(audio_np, initial_prompt=prompt, language='ja',temperature=0.0, beam_size=2, word_timestamps=True )
            seglist:SegList = SegList(segments,info)
            transcribe_time = time.time()-st
            #seglist.dump()

            text,split_sec = seglist.get_text( split_sec )
            text_list.append(text)

        split_idx = int( split_sec*RATE )
        print(f"#Audio {transcribe_time:.3f}s len:{audio_sec:.3f}s,{len(audio_np)} split:{split_sec:.3f}s,{split_idx}")
        if split_idx>0:
            audio_np = audio_np[split_idx:]
        #time.sleep( .2 )
    print("終了しました")
    bot_voice.stop()

def testrun():
    model_size = "tiny" # 39M
    model_size = "base" # 74M
    # model_size = "small" # 461M
    # model_size = "medium" # 769M
    # model_size = "large" # 1550M
    # model_size = "large-v3"

    bot:FasterVoice = FasterVoice(model_size)
    bot.testrun()

if __name__ == "__main__":
    #mlstest()
    #test_seg()
    testrun()