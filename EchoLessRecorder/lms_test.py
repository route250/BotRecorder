
"""
pyaudioで音声を再生しつつ、同時にマイクで録音する。
録音した音声データから、lms適用フィルタを使って再生した音声を除去する。
つまり音声チャットbotが喋った音をフィルタで除去して人間の声だけにして認識できるようにする。
このプログラムでは、音声の再生・録音部分だけ実装する
class LMSAudioには以下の関数をつくる
start() 処理を開始
stop() 停止
add_play(audiodata:np.ndarray) 再生する音声データを追加する。追加した音声はすぐに再生開始
get_audio() 前回getした後で溜まった音声データを取得する
"""