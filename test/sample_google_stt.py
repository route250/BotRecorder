import sys,os
import queue
import pyaudio
from google.cloud import speech

stream_close = False       # ストリーミング終了時にTrueとなる

STREAMING_LIMIT = 240000  
SAMPLE_RATE = 16000
CHUNK_SIZE = int(SAMPLE_RATE / 10)  


class ResumableMicrophoneStream:

    def __init__(self, rate, chunk_size):
        
        self._rate = rate
        self.chunk_size = chunk_size
        self._num_channels = 1
    
        # 取得した音声を格納するキュー
        self._buff = queue.Queue()                 
    
        # マイクから音声を入力するインスタンス
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=self._num_channels,
            rate=self._rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._fill_buffer,
        )

    
    # with文実行時に呼ばれる
    def __enter__(self):

        global stream_close
        stream_close = False
        return self

    # with文終了時に呼ばれる
    def __exit__(self, type, value, traceback):

        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self._buff.put(None)
        self._audio_interface.terminate()
        global stream_close
        stream_close = True

        
    def _fill_buffer(self, in_data, *args, **kwargs):

        # マイクから入力した音声をキューに格納する
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    
    def generator(self):

        global stream_close
        while not stream_close:
            data = []

            chunk = self._buff.get()
            
            if chunk is None:
                return

            data.append(chunk)
            
            # キューが空になるまでdataリストに追加する
            while True:
                try:
                    chunk = self._buff.get(block=False)

                    if chunk is None:
                        return
                    data.append(chunk)

                except queue.Empty:
                    break

            yield b''.join(data)


## 音声のテキスト化を表示する関数
def listen_print_loop(responses, stream):
    
    global stream_close

    for response in responses:

        if not response.results:
            continue

        result = response.results[0]

        if not result.alternatives:
            continue

        transcript = result.alternatives[0].transcript

        # 文末と判定したら区切る
        if result.is_final:
            print(transcript)
        else:
            print('    ', transcript)


        # 『エンド』を言うと終了する
        if transcript == 'エンド':
            stream_close = True
           
            
## Speech to Textを実行する関数    
def excecute_speech_to_text_streaming():

    print('Start Speech to Text Streaming')

    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=SAMPLE_RATE,
        language_code='ja-JP',
        max_alternatives=1
    )
    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True
    )

    mic_manager = ResumableMicrophoneStream(SAMPLE_RATE, CHUNK_SIZE)
    with mic_manager as stream:
           
        # マイクから入力した音声の取得
        audio_generator = stream.generator()

        requests = (
            speech.StreamingRecognizeRequest(audio_content=content) for content in audio_generator
        )

        # Google Speech to Text APIを使って音声をテキストに変換
        responses = client.streaming_recognize(
            streaming_config,
            requests
        )
        
        # テキスト変換結果を表示する
        listen_print_loop(responses, stream)

    print('End Speech to Text Streaming')


if __name__ == '__main__':
    os.environ['GOOGLE_APPLICATION_CREDENTIALS']='/Users/maeda/LLM/BotRecorder/hidden-chiller-434114-t2-4d39fc610ee8.json'
    excecute_speech_to_text_streaming()
