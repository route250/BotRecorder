import sys,os
from pathlib import Path
from dotenv import load_dotenv

from openai.types.completion_usage import CompletionUsage


def setup_openai_api():
    """
    OpenAI APIをセットアップする関数です。
    .envファイルからAPIキーを読み込み、表示します。
    """
    dotenv_path = os.path.join(Path.home(), 'Documents', 'openai_api_key.txt')
    load_dotenv(dotenv_path)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print(f"OPENAI_API_KEY={api_key[:5]}***{api_key[-3:]}")


def update_usage(usage:CompletionUsage|None,update:CompletionUsage|None):
    if usage and update:
        usage.completion_tokens += update.completion_tokens
        usage.prompt_tokens += update.prompt_tokens
        usage.total_tokens += update.total_tokens

def usage_to_price(usage:CompletionUsage|None, *, doll_yen:float=150.0) ->tuple[float,float,float,float]:
    # gpt-4o-mini
    # api
    #  $0.150 / 1M input tokens
    #  $0.600 / 1M output tokens
    # batch
    #  $0.075 / 1M input tokens
    #  $0.300 / 1M output tokens
    if usage:
        in_us:float = usage.prompt_tokens/1000000.0 * 0.15
        out_us:float = usage.completion_tokens/1000000.0 * 0.6
        in_jp:float = in_us*doll_yen
        out_jp:float = out_us*doll_yen
        return in_us,out_us,in_jp,out_jp
    return -1,-1,-1,-1

def usage_to_dict(usage:CompletionUsage|None, *, doll_yen:float=150.0) ->dict:
    in_us,out_us,in_jp,out_jp = usage_to_price(usage,doll_yen=doll_yen)
    result = {}
    if usage and in_us>=0:
        result['tokens'] = {
            'total': usage.total_tokens,
            'in': usage.prompt_tokens,
            'out': usage.completion_tokens
        }
        result['price_jp'] = {
            'total': round( in_jp+out_jp, 2 ),
            'in': round( in_jp, 2 ),
            'out': round( out_jp, 2)
        }
        result['price_us'] = {
            'total': round( in_us+out_us, 4 ),
            'in': round( in_us, 4 ),
            'out': round( out_us, 4 )
        }
    return result

def usage_to_str(usage:CompletionUsage|None, *, doll_yen:float=150.0) ->str:
    in_us,out_us,in_jp,out_jp = usage_to_price(usage,doll_yen=doll_yen)
    if usage and in_us>=0:
        return f"tokens:{usage.total_tokens}({usage.prompt_tokens}+{usage.completion_tokens}) price ${in_us+out_us:.4f} Yen:{in_jp+out_jp:.2f}"
    return ""
