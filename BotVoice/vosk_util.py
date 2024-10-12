import json

IGNORE_WORDS=['小','えー','ん','あ']
NOIZE_WORD='<noize>'
def transcrib_strip(text:str) ->str|None:
    if isinstance(text,str):
        text = text.strip()
        if len(text)>0 and text not in IGNORE_WORDS:
            return text
    return None

def text_strip(text:str) ->str:
    if not isinstance(text,str) or text=="":
        return ""
    tokens:list[str] = text.split()
    res:str = ""
    noize:bool = False
    for t in tokens:
        if t in IGNORE_WORDS:
            if not noize:
                res += NOIZE_WORD
            noize = True
        else:
            res += t
    return res
        
def get_text( vosk_res:dict ) ->str|None:
    #print( json.dumps(vosk_res,ensure_ascii=False))
    if "alternatives" in vosk_res:
        pp = []
        al = vosk_res.get("alternatives",[])
        for x in al:
            con = x.get("confidence",0.0)
            txt = text_strip(x.get("text",""))
            txt = txt if txt!=NOIZE_WORD else ""
            if txt:
                pp.append(txt)
        return "|".join(pp)
    elif "text" in vosk_res:
        text = vosk_res.get("text","")
        text = text_strip(text)
        return text if text!=NOIZE_WORD else ""
    elif "partial" in vosk_res:
        text = vosk_res.get("partial","")
        return text_strip(text)
    return NOIZE_WORD
# 'alternatives' =
# [{'confidence': 208.811737, 'text': ''}, {'confidence': 207.282578, 'text': 'ん'}, {'confidence': 206.940567, 'text': 'えー'}]
# special variables
# function variables
# 0 =
# {'confidence': 208.811737, 'text': ''}
# 1 =
# {'confidence': 207.282578, 'text': 'ん'}
# 2 =
# {'confidence': 206.940567, 'text': 'えー'}
def test():
    data = " 小 小 えー こんにちは"
    res = text_strip(data)
    print(res)

if __name__ == "__main__":
    test()