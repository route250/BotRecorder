import sys, os
import tkinter as tk
from tkinter import filedialog, Label, Button
import matplotlib.pyplot as plt
import numpy as np
import subprocess

# AecResクラスとsave_and_plot関数をインポート
sys.path.append(os.getcwd())
from BotVoice.ace_recorder import AecRecorder, AecRes, nlms_echo_cancel2, plot_aecrec
from BotVoice.rec_util import AudioF32, save_wave, load_wave, audio_info

class AecResGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AecRes Viewer")
        self.last_dir = '/tmp/logdir'
        self.create_widgets()

    def create_widgets(self):
        # ファイル選択ボタン
        self.load_button = Button(self.root, text="Load AecRes File", command=self.load_file)
        self.load_button.pack(pady=10)

        # ファイル名表示ラベル
        self.filename_label = Label(self.root, text="No file selected")
        self.filename_label.pack(pady=10)

        # グラフ表示ボタン
        self.plot_button = Button(self.root, text="Plot Graph", command=self.plot_graph, state=tk.DISABLED)
        self.plot_button.pack(pady=10)

        # 音声再生ボタン
        self.play_buttons = []

    def load_file(self):
        # ファイル選択ダイアログを開く
        filename = filedialog.askopenfilename(initialdir=self.last_dir, title="Select AecRes File", filetypes=[("NPZ files", "*.npz")])
        if filename:
            self.filename = filename
            self.last_dir = os.path.dirname(filename)
            base = os.path.splitext(filename)[0]
            self.rec = AecRes.empty(sampling_rate=16000)  # 初期化用のダミー
            self.rec.load(filename)
            print(f"[INFO] Loaded file: {filename}")
            self.filename_label.config(text=f"Loaded file: {os.path.basename(filename)}")
            self.plot_button.config(state=tk.NORMAL)

            # 音声再生ボタンの作成
            self.create_play_buttons(base)

    def create_play_buttons(self, base):
        # 既存の再生ボタンを削除
        for button in self.play_buttons:
            button.pack_forget()
        self.play_buttons = []

        # 再生ボタンを作成
        for suffix in [".wav", "_lms.wav", "_mic.wav"]:
            audio_file = f"{base}{suffix}"
            if os.path.exists(audio_file):
                button = Button(self.root, text=f"Play {os.path.basename(audio_file)}", command=lambda file=audio_file: self.play_audio(file))
                button.pack(pady=5)
                self.play_buttons.append(button)

    def play_audio(self, filepath):
        # 音声ファイルを再生
        try:
            subprocess.Popen(["afplay", filepath])
            print(f"[INFO] Playing file: {filepath} using afplay")
        except FileNotFoundError:
            print(f"[ERROR] afplay command not found or file not found: {filepath}")

    def plot_graph(self):
        if hasattr(self, 'rec'):
            # グラフを表示
            plot_aecrec(self.rec)
        else:
            print("[ERROR] No AecRes data loaded.")

if __name__ == "__main__":
    root = tk.Tk()
    app = AecResGUI(root)
    root.mainloop()