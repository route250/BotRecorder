import sys,os
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())

from BotVoice.rec_util import AudioF32, EmptyF32, np_append, save_wave, load_wave, signal_ave, sin_signal
from BotVoice.bot_audio import BotAudio,RATE,CHUNK_LEN



def main():

    # 読み込むWaveファイルの設定
    play_filename = 'test/testData/ttsmaker-file-2024-8-27-20-36-50.wav'
    output_raw_filename = 'tmp/raw_audio.wav'

    bot_voice:BotAudio = BotAudio()

    # 再生音をnumpy配列に読み込む
    play_audio_f32:AudioF32 = load_wave( play_filename, sampling_rate=RATE )
    # play_audio_f32 = sin_signal( freq=220, duration=3.0, vol=0.5 )
    
    bot_voice.add_play( play_audio_f32 )

    raw_audio_list = []

    # 指定した期間録音
    print("録音と再生を開始します...")
    bot_voice.start()
    while bot_voice.is_active() and bot_voice.is_playing():
        raw_seg_f32 = bot_voice.get_audio()
        if len(raw_seg_f32)>0:
            raw_audio_list.append(raw_seg_f32)
        time.sleep( 1.5 )  # ミリ秒単位で指定
    print("終了しました")
    bot_voice.stop()

    # 生の録音音声を保存
    raw_seg_f32 = np.concatenate( raw_audio_list )
    save_wave(output_raw_filename, raw_seg_f32, sampling_rate=RATE,ch=1)
    print(f"録音されたデータが {output_raw_filename} に保存されました。")

    ps:int = CHUNK_LEN*5
    pe:int = CHUNK_LEN*6

    plt.figure()
    plt.plot(raw_seg_f32[ps:pe], label='Mic Signal')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    #mlstest()
    main()