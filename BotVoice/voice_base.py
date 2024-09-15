import sys,os
import time
import json
import wave
from io import BytesIO
from typing import Iterable, TypedDict
import numpy as np
import pyaudio

from BotVoice.rec_util import AudioI8, AudioI16, AudioF32, EmptyF32, np_append, save_wave, load_wave, signal_ave, sin_signal
from BotVoice.rec_util import from_f32
from BotVoice.rec_util import as_str, as_list, as_int, as_float
from BotVoice.segments import TranscribRes, Segment, Word
from BotVoice.bot_audio import BotAudio,RATE,CHUNK_SEC,CHUNK_LEN

class VoiceBase:

    def load_model(self):
        pass

    def check_model(self):
        pass

    def transcrib(self, audio:AudioF32 ) ->TranscribRes:
        raise NotImplemented()

    def testrun(self):

        print("Load model")
        self.load_model()
        print("Done")

        print("Check model")
        self.check_model()
        print("Done")

        audio_np:np.ndarray = np.zeros( RATE*3, dtype=np.float32 )
        text_list:list[str] = []
        delta_sec:float = 0.6
        threshold:float = 0.2
        # 指定した期間録音
        print("録音と再生を開始します...")
        start_time:float = time.time()
        stop_time:float = start_time + 180.0

        bot_voice:BotAudio = BotAudio()
        bot_voice._rec_boost
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
            audio_max:float = round(audio_np.max(),3)

            transcribe_time:float = 0.0
            text = ''
            split_sec = audio_sec - delta_sec
            if audio_max>threshold:
                print()
                prompt = ' '.join( text_list[-2:])
                st = time.time()
                seglist:TranscribRes = self.transcrib( audio_np )
                transcribe_time = time.time()-st
                #seglist.dump()

                text,split_sec = seglist.get_text( split_sec )
                text_list.append(text)
                print(f"#Audio {transcribe_time:.3f}s len:{audio_sec:.3f}s,{len(audio_np)} lv:{audio_max}")

            split_idx = int( split_sec*RATE )
            # print(f"#Audio {transcribe_time:.3f}s len:{audio_sec:.3f}s,{len(audio_np)} lv:{audio_max} split:{split_sec:.3f}s,{split_idx}")
            if split_idx>0:
                audio_np = audio_np[split_idx:]
            #time.sleep( .2 )
        print("終了しました")
        bot_voice.stop()
