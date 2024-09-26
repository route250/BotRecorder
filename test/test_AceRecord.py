import sys,os
import time
import traceback

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

import mlx_whisper.whisper
import mlx_whisper

from style_bert_vits2.nlp import bert_models
from style_bert_vits2.constants import Languages
from style_bert_vits2.tts_model import TTSModel

sys.path.append(os.getcwd())
from BotVoice.ace_recorder import AecRecorder, nlms_echo_cancel
from BotVoice.rec_util import AudioF32, save_wave, load_wave, audio_info, sin_signal

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
    recorder.play(playback_data7)
    if mode==1:
        while True:
            time.sleep(0.5)
            print("+",end="")
            delta_lms_f32, delta_mic_f32, delta_spk_f32 = recorder.get_aec_audio()
            mic_f32 = np.concatenate( (mic_f32,delta_mic_f32) )
            lms_f32 = np.concatenate( (lms_f32,delta_lms_f32) )
            spk_f32 = np.concatenate( (spk_f32,delta_spk_f32) )
            if not recorder.is_playing:
                break
        recorder.stop()
    else:
        time.sleep(nsec)
        while recorder.is_playing:
            print("+",end="")
            time.sleep(0.5)
        # 停止
        recorder.stop()
        mic_f32, spk_f32 = recorder.get_raw_audio()
        lms_f32:AudioF32 = nlms_echo_cancel( mic_f32, spk_f32, recorder.aec_mu, recorder.aec_w )

    print("---")
    print(f"[OUT] mic {audio_info(mic_f32,sample_rate=sample_rate)}")
    print(f"[OUT] spk {audio_info(spk_f32,sample_rate=sample_rate)}")

    save_wave( 'tmp/mic_output.wav', mic_f32, sampling_rate=sample_rate, ch=1)
    save_wave( 'tmp/spk_output.wav', spk_f32, sampling_rate=sample_rate, ch=1)

    save_wave( 'tmp/lms_output.wav', lms_f32, sampling_rate=sample_rate, ch=1)

    plt.figure()
    plt.plot(mic_f32, label='Mic')
    plt.plot(spk_f32, label='Raw Spk', alpha=0.2)
    plt.legend()

    plt.figure()
    plt.plot(mic_f32, label='Mic', alpha=0.5)
    plt.plot(lms_f32, label='Lms Mic', alpha=0.5)
    plt.legend()

    plt.show()

    print("")

def w_transcrib(audio_np:AudioF32,model_size):
    st = time.time()
    transcribe_res = mlx_whisper.transcribe( audio_np,no_speech_threshold=0.4,
                        language='ja', word_timestamps=True,
                        fp16=False, path_or_hf_repo=model_size)
    if isinstance(transcribe_res,dict):
        txts = []
        for seg in transcribe_res.get('segments',[]):
            if isinstance(seg,dict):
                txt = seg.get('text','')
                st = seg.get('start')
                ed = seg.get('end')
                avg_logprob = seg.get('avg_logprob',0.0)
                compression_ratio = seg.get('compression_ratio',0.0)
                no_speech_prob = seg.get('no_speech_prob',0.0)
                txts.append(txt)
                print(f"{txt} prob:{avg_logprob} comp:{compression_ratio} no_speech:{no_speech_prob}")
    return ''

def main_x():

    # https://github.com/litagin02/Style-Bert-VITS2/blob/master/library.ipynb

    # BERTモデルをロード（ローカルに手動でダウンロードする必要はありません）

    # model_assetsディレクトリにダウンロードされます
    tmpdir='tmp/model_assets'
    assets_root = Path(tmpdir)

    bert_models.load_model(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm",)
    bert_models.load_tokenizer(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")

    # Hugging Faceから試しにデフォルトモデルをダウンロードしてみて、それを音声合成に使ってみる

    # https://huggingface.co/litagin/style_bert_vits2_jvnv/tree/main

    model_file = "jvnv-F1-jp/jvnv-F1-jp_e160_s14000.safetensors"
    config_file = "jvnv-F1-jp/config.json"
    style_file = "jvnv-F1-jp/style_vectors.npy"

    model_file = "jvnv-M1-jp/jvnv-M1-jp_e158_s14000.safetensors"
    config_file = "jvnv-M1-jp/config.json"
    style_file = "jvnv-M1-jp/style_vectors.npy"

    # model_file = "jvnv-M2-jp/jvnv-M2-jp_e159_s17000.safetensors"
    # config_file = "jvnv-M2-jp/config.json"
    # style_file = "jvnv-M2-jp/style_vectors.npy"

    # model_file = "Rinne/Rinne.safetensors"
    # config_file = "Rinne/config.json"
    # style_file = "Rinne/style_vectors.npy"

    # model_file = "AbeShinzo0708/AbeShinzo20240210_e300_s43800.safetensors"
    # config_file = "AbeShinzo0708/config.json"
    # style_file = "AbeShinzo0708/style_vectors.npy"

    for file in [model_file, config_file, style_file]:
        print(file)
        # hf_hub_download("litagin/style_bert_vits2_jvnv", file, local_dir=str(assets_root))

    model = TTSModel(
        model_path=assets_root / model_file,
        config_path=assets_root / config_file,
        style_vec_path=assets_root / style_file,
        device="cpu",
    )

    greetings = [
        # 基本的な挨拶
        "こんにちは、今日はいい天気ですね。",
        "おはようございます、調子はいかがですか？",
        # 自己紹介と簡単な説明
        "私はAIアシスタントです。何かお手伝いできることはありますか？",
        "私の名前は山田太郎です。よろしくお願いします。",
        # 感情を表すフレーズ
        "すごい！本当に素晴らしいですね。",
        "ああ、これは残念です。でも、次はうまくいくと思います。",
        # 質問形式のセリフ
        "コーヒーと紅茶、どちらが好きですか？",
        "この映画、もう見ましたか？",
        # 日常会話のフレーズ
        "最近、新しい本を読み始めました。とても面白いです。",
        "週末に友達と出かける予定です。どこかおすすめの場所はありますか？",
        # 指示や依頼の表現
        "この書類を明日までに準備してください。",
        "もう一度、最初から説明していただけますか？",
        # 励ましの言葉
        "大丈夫、あなたならきっとできるよ。",
        "頑張ってください。応援しています！",
        # ニュース風のアナウンス
        "本日午後、主要な都市で交通規制が行われます。詳しくはウェブサイトをご覧ください。",
        "新しいテクノロジーが発表され、業界に大きな影響を与えると予想されています。",
    ]

    model_size = 'mlx-community/whisper-small-mlx-q4'

    check_audio:AudioF32 = sin_signal()
    res = w_transcrib(check_audio, model_size)

    pa_chunk_size = 3200
    sample_rate:int = 16000

    recorder:AecRecorder = AecRecorder( device=None, pa_chunk_size=pa_chunk_size, sample_rate=sample_rate)

    mic_f32:AudioF32 = np.zeros( 0, dtype=np.float32 )
    lms_f32:AudioF32 = mic_f32
    spk_f32:AudioF32 = mic_f32

    recorder.start()
    time.sleep(0.5)
    recorder.play_marker()
    for g in greetings[:2]:
        sr, audio_i16 = model.infer(text=g, style_weight=0.0 )

        print(g)

        # 音声データを再生
        recorder.play(audio_i16,sr=sr)

        # 再生が完了するまで待機
        while True:
            time.sleep(0.5)
            print("+",end="")
            delta_lms_f32, delta_mic_f32, delta_spk_f32 = recorder.get_aec_audio()
            mic_f32 = np.concatenate( (mic_f32,delta_mic_f32) )
            lms_f32 = np.concatenate( (lms_f32,delta_lms_f32) )
            spk_f32 = np.concatenate( (spk_f32,delta_spk_f32) )
            if not recorder.is_playing():
                break

    print("---whisper raw mic")
    wres = w_transcrib(mic_f32,model_size)
    print(wres)

    print("---whisper lms mic")
    wres = w_transcrib(lms_f32,model_size)
    print(wres)

    print("---")
    print(f"[OUT] mic {audio_info(mic_f32,sample_rate=sample_rate)}")
    save_wave( 'tmp/mic_output.wav', mic_f32, sampling_rate=sample_rate, ch=1)
    print(f"[OUT] spk {audio_info(spk_f32,sample_rate=sample_rate)}")
    save_wave( 'tmp/spk_output.wav', spk_f32, sampling_rate=sample_rate, ch=1)
    print(f"[OUT] lms {audio_info(lms_f32,sample_rate=sample_rate)}")
    save_wave( 'tmp/lms_output.wav', lms_f32, sampling_rate=sample_rate, ch=1)

    plt.figure()
    plt.plot(mic_f32, label='Mic')
    plt.plot(spk_f32, label='Raw Spk', alpha=0.2)
    plt.legend()

    plt.figure()
    plt.plot(mic_f32, label='Mic', alpha=0.5)
    plt.plot(lms_f32, label='Lms Mic', alpha=0.5)
    plt.legend()

    plt.show()

if __name__ == "__main__":
    main_x()