
import sys,os
from typing import Optional,List,NamedTuple
sys.path.append(os.getcwd())
from BotVoice.rec_util import as_str, as_list, as_int, as_float

PRELIST:list[str] = [ "ご視聴ありがとう"]
INCLIST:list[str] = [ "ご視聴ありがとう"]

def is_accept(text:str|None) ->bool:
    if text is None:
        return False
    for pre in PRELIST:
        if text.startswith(pre):
            return False
    for inc in INCLIST:
        if inc in text:
            return False
    return True

#Wordクラスの項目の説明
#
#	1.	start: float
#	•	単語の開始時間を秒単位で示します。この時間は、音声ストリーム内でこの特定の単語が始まる瞬間を示しています。
#	2.	end: float
#	•	単語の終了時間を秒単位で示します。この時間は、音声ストリーム内でこの特定の単語が終わる瞬間を示しています。
#	3.	word: str
#	•	認識された単語自体を示す文字列。音声から抽出された特定の単語がここに格納されます。
#	4.	probability: float
#	•	単語が正しく認識されたとモデルが確信する確率を示します。値は0から1の範囲で、1に近いほど高い確信度を意味します。この確率は、モデルの信頼度を表し、エラー検出や音声認識の精度評価に使用されます。

class Word(NamedTuple):
    start: float
    end: float
    word: str
    probability: float

def as_word( value:dict ) ->Word:
    return Word(
            start=as_float(value.get('start')),
            end=as_float(value.get('end')),
            word=as_str(value.get('word')),
            probability=as_float(value.get('probability')),
        )

#Segmentクラスの項目の説明
#	1.	id: int セグメントの識別子。
#       各セグメントには一意のIDが付与され、音声全体の中での位置を示します。
#	2.	seek: int
#	•	音声ストリーム内でセグメントが開始される位置をバイト単位で示します。これは、音声ファイル内の特定の位置を指し示すために使用されます。
#	3.	start: float
#	•	セグメントの開始時間を秒単位で示します。この時間は、音声がどこからこのセグメントとして認識されたかを示します。
#	4.	end: float
#	•	セグメントの終了時間を秒単位で示します。この時間は、セグメントが音声のどこまでをカバーしているかを示します。
#	5.	text: str
#	•	セグメントから認識されたテキスト。このフィールドには、音声セグメントがテキストに変換された結果が格納されます。
#	6.	tokens: List[int]
#	•	セグメントのテキストに対応するトークンのリスト。各トークンは、音声認識モデルが使用する語彙に基づく数値表現です。
#	7.	temperature: float デコード時に使用されたtemperatureパラメータ。
#       temperatureは、生成されるテキストの確率分布の多様性を制御します。
#       temperatureが高いと、生成されるテキストの多様性が増え、逆にtemperatureが低いと、生成されるテキストはより決定論的になります。
#	8.	avg_logprob: float セグメント全体の平均ログ確率。
#       この値は、認識されたテキストの確率を示し、値が高いほど信頼性が高いと考えられます。
#	9.	compression_ratio: float セグメントのテキストの圧縮率。
#       元の音声データに対してどれだけ効率的にテキストが圧縮されているかを示します。
#       圧縮率が高すぎる場合、認識結果が意味をなさない可能性があるため、品質の指標として使用されます。
#	10.	no_speech_prob: float セグメントが無音（または音声がない）である確率。
#       この値が高い場合、そのセグメントが実際には音声を含んでいない可能性があることを示唆します。
#	11.	words: Optional[List[Word]]	セグメント内の各単語に関する詳細情報のリスト。
#       このリストには、Wordオブジェクトが含まれており、各単語の開始時間、終了時間、およびテキストが格納されています。
#       音声認識の結果をより詳細に解析するために使用されます。
class Segment(NamedTuple):
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: List[int]
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float
    words: Optional[List[Word]]

    def get_word_start(self) ->float:
        if self.words and len(self.words)>0:
            return self.words[0].start
        else:
            return self.start

    def get_word_end(self) ->float:
        if self.words and len(self.words)>0:
            return self.words[-1].end
        else:
            return self.end

def as_segment( data ) ->Segment:
    return Segment(
        id = as_int(data.get('id')),
        seek = as_int(data.get('seek')),
        start = as_float(data.get('start')),
        end = as_float(data.get('end')),
        text = as_str(data.get('text')),
        tokens = [],
        temperature=as_float(data.get('temperature')),
        avg_logprob=as_float(data.get('avg_logprob')),
        compression_ratio=as_float(data.get('compression_ratio')),
        no_speech_prob=as_float(data.get('no_speech_prob')),
        words = [ as_word(w) for w in as_list(data.get('words')) ]
    )

def get_gap( a:Segment, b:Segment ) ->float:
    return b.get_word_start() - a.get_word_end()

def join_segment( a:Segment, b:Segment, gap:float ) ->Segment|None:
    dist:float = get_gap(a,b)
    if dist<0.0 or gap<dist:
        return None
    words:Optional[List[Word]] = None
    if a.words or b.words:
        words = ( a.words or [] ) + ( b.words or [] )
    return Segment(
        id = a.id,
        seek = a.seek,
        start = a.start,
        end = b.end,
        text = a.text + b.text,
        tokens = a.tokens + b.tokens,
        temperature=a.temperature,
        avg_logprob=(a.avg_logprob+b.avg_logprob)*0.5,
        compression_ratio=max(a.compression_ratio,b.compression_ratio),
        no_speech_prob=max(a.no_speech_prob,b.no_speech_prob),
        words=words
    )

def split_segment0( seg:Segment, gap:float ) ->tuple[Segment|None,Segment|None]:
    if not seg.words:
        return None,None
    split:int = len(seg.words)
    for i in range(1,len(seg.words)):
        if seg.words[i].start-seg.words[i-1].end>=gap:
            split = i
            break
    if split>=len(seg.words):
        return None,None
    a:Segment = Segment(
        id=seg.id,
        seek=seg.seek,
        start=seg.start,
        end=seg.words[split-1].end,
        text=''.join( [w.word for w in seg.words[:split]]),
        tokens=seg.tokens,
        temperature=seg.temperature,
        avg_logprob=seg.avg_logprob,
        compression_ratio=seg.compression_ratio,
        no_speech_prob=seg.no_speech_prob,
        words=seg.words[:split]
    )
    b:Segment = Segment(
        id=seg.id,
        seek=seg.seek,
        start=seg.words[split].start,
        end=seg.end,
        text=''.join( [w.word for w in seg.words[split:]]),
        tokens=seg.tokens,
        temperature=seg.temperature,
        avg_logprob=seg.avg_logprob,
        compression_ratio=seg.compression_ratio,
        no_speech_prob=seg.no_speech_prob,
        words=seg.words[split:]
    )
    return a,b

def split_segment(seg:Segment,gap:float) ->list[Segment]:
    res:list[Segment] = []
    while True:
        a,b = split_segment0(seg,gap)
        if a and b:
            res.append(a)
            seg = b
        else:
            res.append(seg)
            break
    return res

def merge_segment( seg_list:list[Segment],gap:float) ->list[Segment]:
    res:list[Segment] = split_segment(seg_list[0],gap)
    for seg in seg_list[1:]:
        sub:list[Segment] = split_segment(seg,gap)
        a = join_segment(res[-1],sub[0],gap)
        if a:
            res = res[:-1] + [a] + sub[1:]
        else:
            res.extend( sub )
    return res

class TranscribRes:
    def __init__(self,data:dict[str,str|list]):
        self.text:str = as_str(data.get('text'))
        seg_list = []
        for seg in as_list(data.get('segments')):
            if seg.get('no_speech_prob',1.0)<0.9 and is_accept(seg.get('text')):
                seg_list.append( as_segment(seg) )
        self.segments:list[Segment] = seg_list
        self.language:str = as_str(data.get('language'))
        self.transcribe_time:float = 0.0

    def get_text(self, end_sec:float ) ->tuple[str,float]:
        print("# get_text")
        split_sec:float = 0
        for seg in self.segments:
            if seg.words:
                for word in seg.words:
                    if float(word.end)<=end_sec:
                        split_sec=float(seg.end)
        if split_sec == 0.0:
            return '',end_sec
        text_list = []
        for idx,seg in enumerate(self.segments):
            if float(seg.end)<=split_sec:
                print(f"  {idx:2d} [{seg.start:.2f}-{seg.end:.2f}] {seg.no_speech_prob:.2f} {seg.text}")
                if seg.words:
                    for iw,word in enumerate(seg.words):
                        print(f"  {idx:2d}-{iw:2d} [{word.start:.2f}-{word.end:.2f}] {word.word}")
                text_list.append(seg.text)
            elif float(seg.start)<split_sec:
                print(f"  {idx:2d} [{seg.start:.2f}-{seg.end:.2f}] {seg.no_speech_prob:.2f} ")
                ww = []
                if seg.words:
                    for iw,word in enumerate(seg.words):
                        if float(word.end)<=split_sec:
                            print(f"  {idx:2d}-{iw:2d} [{word.start:.2f}-{word.end:.2f}] {word.word}")
                            text_list.append(word.word)
                        else:
                            print(f"  {idx:2d}-{iw:2d} [{word.start:.2f}-{word.end:.2f}] ### {word.word}")
                text_list.append(''.join(ww))
            else:
                print(f"  {idx:2d} [{seg.start:.2f}-{seg.end:.2f}] ### {seg.text}")
        return ' '.join(text_list), split_sec

    def dump(self):
        print("# Segments")
        for s in self.segments:
            print(f"[{s.start:.3f}-{s.end:.3f}] {s.no_speech_prob:.3f} {s.text}")