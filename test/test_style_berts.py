
from pathlib import Path
import numpy as np
from huggingface_hub import hf_hub_download
import sounddevice as sd
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.constants import Languages
from style_bert_vits2.tts_model import TTSModel

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
    "えー？", "えーー？",
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

for g in greetings:
    sr, audio = model.infer(text=g, style_weight=0.0 )

    print(g)
    # 音声データを再生
    sd.play(audio, samplerate=sr)

    # 再生が完了するまで待機
    sd.wait()
