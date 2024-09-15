# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Google Cloud Speech API sample application using the streaming API.

NOTE: This module requires the dependencies `pyaudio` and `termcolor`.
To install using pip:

    pip install pyaudio
    pip install termcolor

Example usage:
    python transcribe_streaming_infinite.py
"""

# [START speech_transcribe_infinite_streaming]
import sys,os
import queue
import re
import sys
import time
from typing import Iterable, Iterator, Generator,Any
import json

from google.cloud import speech
from google.cloud.speech_v1.types import StreamingRecognizeRequest,StreamingRecognizeResponse
import pyaudio

# Audio recording parameters
STREAMING_LIMIT = 240000  # 4 minutes
SAMPLE_RATE = 16000
CHUNK_SIZE = int(SAMPLE_RATE / 10)  # 100ms

ERACE = "\033[K"
RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[0;33m"


def get_current_time() -> int:
    """Return Current Time in MS.

    Returns:
        int: Current Time in MS.
    """

    return int(round(time.time() * 1000))


class ResumableMicrophoneStream:
    """Opens a recording stream as a generator yielding the audio chunks."""

    def __init__(
        self,
        rate: int,
        chunk_size: int,
    ) -> None:
        """Creates a resumable microphone stream.

        Args:
        self: The class instance.
        rate: The audio file's sampling rate.
        chunk_size: The audio file's chunk size.

        returns: None
        """
        self._rate:int = rate
        self.chunk_size:int = chunk_size
        self._num_channels:int = 1
        self._buff:queue.Queue = queue.Queue()
        self.closed:bool = True
        self.start_time = get_current_time()
        self.restart_counter:int = 0
        self.audio_input = []
        self.last_audio_input = []
        self.final_request_end_time = 0
        self.bridging_offset = 0
        self._new_stream = True
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=self._num_channels,
            rate=self._rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            # Run the audio stream asynchronously to fill the buffer object.
            # This is necessary so that the input device's buffer doesn't
            # overflow while the calling thread makes network requests, etc.
            stream_callback=self._fill_buffer,
        )

    def __enter__(self):
        """Opens the stream.

        Args:
        self: The class instance.

        returns: None
        """
        self.closed = False
        return self

    def __exit__(
        self,
        type: object,
        value: object,
        traceback: object,
    ):
        """Closes the stream and releases resources.

        Args:
        self: The class instance.
        type: The exception type.
        value: The exception value.
        traceback: The exception traceback.

        returns: None
        """
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(
        self,
        in_data: object,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Continuously collect data from the audio stream, into the buffer.

        Args:
        self: The class instance.
        in_data: The audio data as a bytes object.
        args: Additional arguments.
        kwargs: Additional arguments.

        returns: None
        """
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        """Stream Audio from microphone to API and to local buffer

        Args:
            self: The class instance.

        returns:
            The data from the audio stream.
        """
        while not self.closed:
            data = []

            if self._new_stream and self.last_audio_input:
                chunk_time = STREAMING_LIMIT / len(self.last_audio_input)

                if chunk_time != 0:
                    if self.bridging_offset < 0:
                        self.bridging_offset = 0

                    if self.bridging_offset > self.final_request_end_time:
                        self.bridging_offset = self.final_request_end_time

                    chunks_from_ms = round(
                        (self.final_request_end_time - self.bridging_offset)
                        / chunk_time
                    )

                    self.bridging_offset = round(
                        (len(self.last_audio_input) - chunks_from_ms) * chunk_time
                    )

                    for i in range(chunks_from_ms, len(self.last_audio_input)):
                        data.append(self.last_audio_input[i])

                self._new_stream = False

            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            self.audio_input.append(chunk)

            if chunk is None:
                return
            data.append(chunk)
            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)

                    if chunk is None:
                        return
                    data.append(chunk)
                    self.audio_input.append(chunk)

                except queue.Empty:
                    break

            yield b"".join(data)

class AudioSubIterator(Iterable[StreamingRecognizeRequest]):
    """オーディオを区切った区間のイテレータ"""
    def __init__(self,iter):
        self.iter = iter
        self.stop:bool = False
    def __iter__(self):
        return self
    def __next__(self):
        try:
            if self.iter and not self.stop:
                ret = self.iter.__next__()
                if ret is not None:
                    return ret
        except:
            pass
        self.iter=None
        raise StopIteration()

class AudioMainIterator(Iterable[AudioSubIterator]):
    """オーディオ入力全体のイテレータ"""
    def __init__(self, mic_manager:ResumableMicrophoneStream ):
        self.stream:ResumableMicrophoneStream = mic_manager
        self._iter = self._request_generator()
        self._closed:bool = False

    def _request_generator(self) ->Generator[StreamingRecognizeRequest,None,None]:
        with self.stream as stream:
            while not stream.closed:
                audio_generator = stream.generator()
                if audio_generator:
                    for content in audio_generator:
                        if content is not None:
                            yield StreamingRecognizeRequest(audio_content=content)
        self._closed = True
        print(f"[Mic]Closed")

    def __iter__(self):
        return self

    def __next__(self):
        if self._closed:
            raise StopIteration()
        #print(':',end="")
        return AudioSubIterator(self._iter)

class RecgIterator(Iterable[StreamingRecognizeResponse]):
    """音声認識結果のイテレータ"""
    def __init__(self, mic_manager:ResumableMicrophoneStream):
        self.stream:ResumableMicrophoneStream = mic_manager
        self.client =speech.SpeechClient()
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=SAMPLE_RATE,
            language_code="ja-JP",
            max_alternatives=1,
            model='command_and_search',use_enhanced=True,
            enable_automatic_punctuation=True,  # 句読点の自動挿入を有効にする
        )
        self.streaming_config = speech.StreamingRecognitionConfig(
            config=config, interim_results=True,single_utterance=True
        )

    def __iter__(self):
        return self.ResStream()
    
    def ResStream(self) ->Generator[StreamingRecognizeResponse,None,None]:
        full_iter:AudioMainIterator = AudioMainIterator(self.stream)
        for sub_iter in full_iter:
            res_iter:Iterator[StreamingRecognizeResponse] = self.client.streaming_recognize(self.streaming_config, sub_iter )
            for response in res_iter:
                eos = response.speech_event_type == StreamingRecognizeResponse.SpeechEventType.END_OF_SINGLE_UTTERANCE
                is_final = response.results[0].is_final if len(response.results)>0 else False
                if eos:
                    sub_iter.stop=True
                # if is_final:
                #     print("<Final>",end="")
                # elif eos:
                #     print("<EOS>",end="")
                # else:
                #     print(".",end="")
                yield response
            #print(f"\r{ERACE}",end="")

def main() -> None:
    """start bidirectional streaming from microphone input to speech API"""

    mic_stream:ResumableMicrophoneStream = ResumableMicrophoneStream(SAMPLE_RATE, CHUNK_SIZE)

    sys.stdout.write(YELLOW)
    sys.stdout.write('\nListening, say "Quit" or "Exit" or "終了" to stop.\n\n')
    sys.stdout.write("End (ms)       Transcript Results/Status\n")
    sys.stdout.write("=====================================================\n")

    recg_iter:RecgIterator = RecgIterator(mic_stream)
    for response in recg_iter:

        if not response.results:
            continue

        result = response.results[0]

        if not result.alternatives:
            continue

        transcript = result.alternatives[0].transcript
        print("\r",end="")

        result_seconds = 0
        result_micros = 0

        if result.result_end_time.seconds:
            result_seconds = result.result_end_time.seconds

        if result.result_end_time.microseconds:
            result_micros = result.result_end_time.microseconds

        result_end_time = int((result_seconds * 1000) + (result_micros / 1000))

        corrected_time = (
            result_end_time
            - mic_stream.bridging_offset
            + (STREAMING_LIMIT * mic_stream.restart_counter)
        )
        # Display interim results, but with a carriage return at the end of the
        # line, so subsequent lines will overwrite them.
        if result.is_final:
            sys.stdout.write(GREEN)
            sys.stdout.write(ERACE)
            sys.stdout.write(str(corrected_time) + ": " + transcript + "\n")

            # Exit recognition if any of the transcribed phrases could be
            # one of our keywords.
            if re.search(r"\b(exit|quit|終了)\b", transcript, re.I):
                sys.stdout.write(YELLOW)
                sys.stdout.write("Exiting...\n")
                mic_stream.closed = True
                break
        else:
            sys.stdout.write(RED)
            sys.stdout.write(ERACE)
            sys.stdout.write(str(corrected_time) + ": " + transcript + "\r")

if __name__ == "__main__":
    os.environ['GOOGLE_APPLICATION_CREDENTIALS']='/Users/maeda/LLM/BotRecorder/hidden-chiller-434114-t2-4d39fc610ee8.json'
    main()

# [END speech_transcribe_infinite_streaming]
