import numpy as np
from numpy.typing import NDArray

delta = 2
class Tx:
    def __init__(self):
        self.rec:list[int] = []
        self.echo:list[int] = []
        self.cnt:int = 1003

    def callback(self,val:int):
        self.rec.append(val)
        self.echo.append(100+val+delta)
        self.cnt+=1
    def get(self):

        es:int = delta*2
        if len(self.echo)<es:
            return [],[]
        aa = self.rec
        self.rec = []
        bb = self.echo
        self.echo = self.echo[-es:]
        return aa,bb

tx=Tx()
v:int=1
for i in range(10):
    tx.callback(v)
    v+=1

print(f"rec :{tx.rec}")
print(f"echo:{tx.echo}")
aa,bb = tx.get()
print(f"aa:{aa}")
print(f"bb:{bb}")

print("-----------------------------")
for i in range(10):
    tx.callback(v)
    v+=1

print(f"rec :{tx.rec}")
print(f"echo:{tx.echo}")
print("")
aa,bb = tx.get()
print(f"aa:{aa}")
print(f"bb:{bb}")


