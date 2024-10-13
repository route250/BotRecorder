import sys,os
import time
import traceback

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())
from BotVoice.ace_recorder import AecRecorder, AecRes, nlms_echo_cancel2, plot_aecrec
from BotVoice.rec_util import AudioF32, save_wave, load_wave, audio_info

#----------------------
# トーン
#----------------------

def main_get():
   
    #list_microphones()

    pa_chunk_size = 3200
    sample_rate:int = 16000

    mode = 1

    recorder:AecRecorder = AecRecorder( device=None, pa_chunk_size=pa_chunk_size, sample_rate=sample_rate)


    # is_maker_tone( canceller.marker_tone, sample_rate )

    recorder.start()
    time.sleep(1.0)

    playback_data7:AudioF32 = load_wave('test/testData/tts/sample_voice.wav',sampling_rate=sample_rate )
    playback_data7 *= 0.3
    nsec:float = len(playback_data7)/sample_rate

    logdata:AecRes = AecRes.empty(sample_rate)
    _,_ = recorder.get_raw_audio()
    recorder.play_marker() #
    recorder.play('test',playback_data7)
    if mode==1:
        title='seg'
        while True:
            time.sleep(0.5)
            print("+",end="")
            rec:AecRes = recorder.get_aec_audio()
            logdata += rec
            if not recorder.is_playing():
                break
        recorder.stop()

    else:
        time.sleep(nsec)
        while recorder.is_playing():
            print("+",end="")
            time.sleep(0.5)
        # 停止
        recorder.stop()
        time.sleep(0.5)
        logdata = recorder.get_aec_audio()
        title='full'
        print("---")

    filename=f"tmp/out_aec_{title}.npz"
    plot_aecrec( logdata, filename=filename, show=True )

def main_file():

    pa_chunk_size = 3200
    sample_rate:int = 16000

    rec:AecRecorder = AecRecorder( device=None, pa_chunk_size=pa_chunk_size, sample_rate=sample_rate)

    mic_f32:AudioF32 = load_wave('test/testData/aec/aec_mic_input.wav', sampling_rate=16000)
    spk_f32:AudioF32 = load_wave('test/testData/aec/aec_spk_input.wav', sampling_rate=16000)

    logdata:AecRes = rec.convert_aec_audio(mic_f32,spk_f32)
    plot_aecrec( logdata, show=True )

if __name__ == "__main__":
    #main_tome()
    #main()
    # main_get()
    main_file()