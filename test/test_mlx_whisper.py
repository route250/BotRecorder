import sys,os
import time

import mlx_whisper.whisper
import mlx_whisper

sys.path.append(os.getcwd())

from BotVoice.rec_util import AudioI8, AudioI16, AudioF32, EmptyF32, np_append, save_wave, load_wave, signal_ave, sin_signal
from BotVoice.rec_util import from_f32
from BotVoice.rec_util import as_str, as_list, as_int, as_float
from BotVoice.segments import TranscribRes, Segment, Word
from BotVoice.bot_audio import BotAudio,RATE,CHUNK_SEC,CHUNK_LEN
from BotVoice.voice_base import VoiceBase

class MlxVoice(VoiceBase):

    def __init__(self, model:str):
        self.model_size = model

    def load_model(self):
        pass

    def check_model(self):
        check_audio:AudioF32 = sin_signal()
        res:TranscribRes = self.transcrib(check_audio)

    def transcrib(self, audio_np:AudioF32) ->TranscribRes:
        st = time.time()
        transcribe_res = mlx_whisper.transcribe( audio_np,no_speech_threshold=0.4,
                            language='ja', word_timestamps=True,
                            fp16=False, path_or_hf_repo=self.model_size)
        seglist:TranscribRes = TranscribRes(transcribe_res)
        seglist.transcribe_time = time.time()-st
        return seglist

def main():

    # model_size = 'mlx-community/whisper-tiny-mlx'
    # model_size = 'mlx-community/whisper-tiny-mlx-fp32'
    # model_size = 'mlx-community/whisper-tiny-mlx-q4'
    # model_size = 'mlx-community/whisper-tiny-mlx-8bit'

    # model_size = 'mlx-community/whisper-base-mlx'
    # model_size = 'mlx-community/whisper-base-mlx-fp32'
    # model_size = 'mlx-community/whisper-base-mlx-q4'
    # model_size = 'mlx-community/whisper-base-mlx-8bit'
    # model_size = 'mlx-community/whisper-base-mlx-4bit'
    # model_size = 'mlx-community/whisper-base-mlx-2bit'

    # model_size = 'mlx-community/whisper-small-mlx'
    # model_size = 'mlx-community/whisper-small-mlx-fp32'
    # model_size = 'mlx-community/whisper-small-mlx-q4' # 197M
    # model_size = 'mlx-community/whisper-small-mlx-8bit'
    # model_size = 'mlx-community/whisper-small-mlx-4bit'


    # model_size = 'mlx-community/whisper-medium-mlx' # 1.5GB
    # model_size = 'mlx-community/whisper-medium-mlx-fp32' # 3GB
    # model_size = 'mlx-community/whisper-medium-mlx-q4' # 0.5GB
    # model_size = 'mlx-community/whisper-medium-mlx-8bit' # 865M

    # model_size = 'mlx-community/whisper-large-v3-mlx'

    model_size = 'mlx-community/whisper-small-mlx-q4'

    bot:MlxVoice = MlxVoice( model=model_size )
    bot.testrun()

if __name__ == "__main__":
    main()