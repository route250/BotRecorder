import sys,os
import time

import torch
import whisper as openai_whisper
from whisper.model import Whisper as OpenAI_Whisper

sys.path.append(os.getcwd())

from BotVoice.rec_util import AudioI8, AudioI16, AudioF32, EmptyF32, np_append, save_wave, load_wave, signal_ave, sin_signal
from BotVoice.rec_util import from_f32
from BotVoice.rec_util import as_str, as_list, as_int, as_float
from BotVoice.segments import TranscribRes, Segment, Word
from BotVoice.bot_audio import BotAudio,RATE,CHUNK_SEC,CHUNK_LEN
from BotVoice.voice_base import VoiceBase

class OpenaiVoice(VoiceBase):

    def __init__(self, model:str):
        self.model:OpenAI_Whisper
        self.model_size = model

    def load_model(self):
        # モデルの読み出し（今回はsmallモデルを利用）
        self.model = openai_whisper.load_model(self.model_size,device='cpu')
        # if torch.backends.mps.is_available():
        #     m = self.model.half()

    def check_model(self):
        check_audio:AudioF32 = sin_signal()
        res:TranscribRes = self.transcrib(check_audio)

    def transcrib(self, audio_np:AudioF32) ->TranscribRes:
        st = time.time()
        transcribe_res = self.model.transcribe( audio_np,no_speech_threshold=0.4,
                            language='ja', word_timestamps=True,fp16=True)
        seglist:TranscribRes = TranscribRes(transcribe_res)
        seglist.transcribe_time = time.time()-st
        return seglist

def main():

    model_size = 'tiny'

    bot:OpenaiVoice = OpenaiVoice( model=model_size )
    bot.testrun()

if __name__ == "__main__":
    main()