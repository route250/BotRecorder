import sys,os
import time
import traceback

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())
from BotVoice.ace_recorder import AecRecorder, AecRes, nlms_echo_cancel2
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

    mic_f32:AudioF32 = np.zeros( 0, dtype=np.float32 )
    lms_f32:AudioF32 = mic_f32
    spk_f32:AudioF32 = mic_f32
    mask_f32:AudioF32 = mic_f32
    vad_f32:AudioF32 = mic_f32
    errors_f32:AudioF32 = mic_f32

    _,_ = recorder.get_raw_audio()
    recorder.play_marker() #
    recorder.play('test',playback_data7)
    if mode==1:
        title='seg'
        while True:
            time.sleep(0.5)
            print("+",end="")
            rec:AecRes = recorder.get_aec_audio()
            mic_f32 = np.concatenate( (mic_f32,rec.raw) )
            lms_f32 = np.concatenate( (lms_f32,rec.audio) )
            if len(spk_f32)==0:
                spk_f32 = np.concatenate( (spk_f32,rec.spk) )
            else:
                spk_f32 = np.concatenate( (spk_f32,rec.spk[-len(rec.audio):]) )
            mask_f32 = np.concatenate( (mask_f32,rec.mask) )
            vad_f32 = np.concatenate( (vad_f32,rec.vad) )
            errors_f32 = np.concatenate( (errors_f32,rec.errors) )
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
        lms_f32, mask_f32, errors_f32 = nlms_echo_cancel2( mic_f32, spk_f32, recorder.aec_mu, recorder.aec_w )
        title='full'
        print("---")

    save_and_plot( title, mic_f32, spk_f32, lms_f32, mask_f32, vad_f32, errors_f32, sample_rate )

def save_and_plot( title:str, mic_f32:AudioF32, spk_f32:AudioF32,lms_f32:AudioF32, mask_f32:AudioF32, vad:AudioF32, errors_f32:AudioF32, sample_rate ):

    print(f"[OUT] mic {audio_info(mic_f32,sample_rate=sample_rate)}")
    save_wave( f'tmp/out_aec_{title}_mic.wav', mic_f32, sampling_rate=sample_rate, ch=1)

    print(f"[OUT] spk {audio_info(spk_f32,sample_rate=sample_rate)}")
    save_wave( f'tmp/out_aec_{title}_spk.wav', spk_f32, sampling_rate=sample_rate, ch=1)

    print(f"[OUT] lms {audio_info(lms_f32,sample_rate=sample_rate)}")
    save_wave( f'tmp/out_aec_{title}_lms.wav', lms_f32, sampling_rate=sample_rate, ch=1)

    max_y = round( 0.05 + max( np.max(np.abs(mic_f32)), np.max(np.abs(spk_f32)), np.max(np.abs(lms_f32)) ), 1 )
    x1 = np.array( range(len(mic_f32)) )
    x2 = np.array( range(len(spk_f32))) - (len(mic_f32)-len(spk_f32))
    mask_bool = mask_f32>0.0
    plt.figure(figsize=(12,3))
    plt.plot(x1,mic_f32, label='Mic')
    plt.plot(x2,spk_f32, label='Spk', alpha=0.2)
    plt.ylim(-max_y,max_y)
    plt.legend()
    plt.savefig(f'tmp/out_aec_{title}_spk.png',dpi=300)

    # 図の作成
    fig, ax1 = plt.subplots(figsize=(12, 3))
    # Y1軸 (Mic と Lms Mic) にデータをプロット
    ax1.fill_between(x1, -max_y, max_y, where=mask_bool, color='green', lw=0, alpha=0.1, label='mask' ) 
    ax1.plot(x1,mic_f32, label='Mic', alpha=0.3)
    ax1.plot(x1,lms_f32, label='LMS', alpha=0.3)
    ax1.set_ylim(-max_y,max_y)
    ax1.legend(loc='upper left')
    # Y2軸 Maskをプロット
    ax2 = ax1.twinx()    # 画像を保存
    ax2.plot(x1,vad, color='red', label='vad', alpha=0.1)
    ax2.set_ybound(0.0,1.0)
    ax2.set_ylabel('vad')
    ax2.legend(loc='upper right')
    # 画像を保存
    plt.savefig(f'tmp/out_aec_{title}_lms.png', dpi=300)

    # 図の作成
    fig, ax1 = plt.subplots(figsize=(12, 3))
    # Y1軸 Errorsをプロット
    ax1.plot(x1,errors_f32, color='red', label='Errors')
    ax1.set_ylabel('Errors')
    ax1.legend(loc='upper left')
    # Y2軸 Maskをプロット
    ax2 = ax1.twinx()
    ax2.fill_between(x1, 0, mask_f32, color='green', lw=0, alpha=0.2, label='mask' ) 
    ax2.set_ybound(0.0,1.0)
    ax2.set_ylabel('Mask')
    ax2.legend(loc='upper right')
    # 画像を保存
    plt.savefig(f'tmp/out_aec_{title}_errors.png', dpi=300)

    # グラフを表示
    plt.show()

    print("")

def main_file():

    pa_chunk_size = 3200
    sample_rate:int = 16000

    rec:AecRecorder = AecRecorder( device=None, pa_chunk_size=pa_chunk_size, sample_rate=sample_rate)

    mic_f32:AudioF32 = load_wave('test/testData/aec_mic_input.wav', sampling_rate=16000)
    spk_f32:AudioF32 = load_wave('test/testData/aec_spk_input.wav', sampling_rate=16000)

    lms_f32, mask, errors = nlms_echo_cancel2( mic_f32, spk_f32, rec.aec_mu, rec.aec_w )
    vad = errors
    save_and_plot( 'test', mic_f32, spk_f32, lms_f32, mask, vad, errors, sample_rate )

if __name__ == "__main__":
    #main_tome()
    #main()
    main_get()
    # main_file()