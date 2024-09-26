import sys,os
import time
import traceback

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())
from BotVoice.ace_recorder import AecRecorder, nlms_echo_cancel
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

    playback_data7:AudioF32 = load_wave('test/testData/sample_voice.wav',sampling_rate=sample_rate )
    playback_data7 *= 0.3
    nsec:float = len(playback_data7)/sample_rate

    mic_f32:AudioF32 = np.zeros( 0, dtype=np.float32 )
    lms_f32:AudioF32 = mic_f32
    spk_f32:AudioF32 = mic_f32

    _,_ = recorder.get_raw_audio()
    recorder.play_marker()
    recorder.play(playback_data7)
    if mode==1:
        while True:
            time.sleep(0.5)
            print("+",end="")
            delta_lms_f32, delta_mic_f32, delta_spk_f32 = recorder.get_aec_audio()
            mic_f32 = np.concatenate( (mic_f32,delta_mic_f32) )
            lms_f32 = np.concatenate( (lms_f32,delta_lms_f32) )
            spk_f32 = np.concatenate( (spk_f32,delta_spk_f32) )
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
        mic_f32, spk_f32 = recorder.get_raw_audio()
        lms_f32:AudioF32 = nlms_echo_cancel( mic_f32, spk_f32, recorder.aec_mu, recorder.aec_w )

    save_and_plot( mic_f32, spk_f32, lms_f32, sample_rate )

def save_and_plot( mic_f32:AudioF32, spk_f32:AudioF32,lms_f32:AudioF32, sample_rate ):
    print("---")
    print(f"[OUT] mic {audio_info(mic_f32,sample_rate=sample_rate)}")
    save_wave( 'tmp/mic_output.wav', mic_f32, sampling_rate=sample_rate, ch=1)

    print(f"[OUT] spk {audio_info(spk_f32,sample_rate=sample_rate)}")
    save_wave( 'tmp/spk_output.wav', spk_f32, sampling_rate=sample_rate, ch=1)

    print(f"[OUT] lms {audio_info(lms_f32,sample_rate=sample_rate)}")
    save_wave( 'tmp/lms_output.wav', lms_f32, sampling_rate=sample_rate, ch=1)

    plt.figure()
    plt.plot(mic_f32, label='Mic')
    plt.plot(spk_f32[-len(mic_f32):], label='Raw Spk', alpha=0.2)
    plt.legend()
    plt.savefig('tmp/mic_spk_plot.png')

    plt.figure()
    plt.plot(mic_f32, label='Mic', alpha=0.5)
    plt.plot(lms_f32, label='Lms Mic', alpha=0.5)
    plt.legend()
    plt.savefig('tmp/mic_lms_plot.png')
    plt.show()

    print("")

def main_file():

    pa_chunk_size = 3200
    sample_rate:int = 16000

    rec:AecRecorder = AecRecorder( device=None, pa_chunk_size=pa_chunk_size, sample_rate=sample_rate)

    mic_f32:AudioF32 = load_wave('test/testData/aec_mic_input.wav', sampling_rate=16000)
    spk_f32:AudioF32 = load_wave('test/testData/aec_spk_input.wav', sampling_rate=16000)

    lms_f32:AudioF32 = nlms_echo_cancel( mic_f32, spk_f32, rec.aec_mu, rec.aec_w )
    save_and_plot( mic_f32, spk_f32, lms_f32, sample_rate )

if __name__ == "__main__":
    #main_tome()
    #main()
    main_get()
    # main_file()