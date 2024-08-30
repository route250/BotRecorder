import sys,os

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())
from BotVoice.echo_less_recorder import AudioF32, sin_signal

def main():
    data, seg_f32 = sin_signal( freq=220, duration=1.0, vol=1.0 )

    plt.figure()
    plt.plot(data, label='Signal')
    plt.legend()

    plt.figure()
    plt.plot(seg_f32, label='Segment')
    plt.legend()

    plt.show()

if __name__ == "__main__":
    main()