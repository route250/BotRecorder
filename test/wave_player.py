import tkinter as tk
from tkinter import filedialog, messagebox, Scrollbar
import sounddevice as sd
import wave
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import threading
import time
from matplotlib.widgets import Slider

matplotlib.use('TkAgg')


class AudioPlayer:
    def __init__(self, master):
        self.master = master
        self.master.title("WAV Player")

        # Waveform data
        self.data = None
        self.fs = None
        self.duration = 0
        self.stream = None
        self.play_thread = None
        self.is_playing = False
        self.stop_flag = False
        self.start_idx = 0
        self.end_idx = None
        self.zoom_factor = 1.0
        self.offset = 0

        # GUI Components
        self.create_widgets()

    def create_widgets(self):
        # Button frame
        frame = tk.Frame(self.master)
        frame.pack(side=tk.TOP, pady=10)

        self.load_button = tk.Button(frame, text="Load", command=self.load_file)
        self.load_button.pack(side=tk.LEFT, padx=5)

        self.play_button = tk.Button(frame, text="Play", command=self.play_audio, state=tk.DISABLED)
        self.play_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = tk.Button(frame, text="Stop", command=self.stop_audio, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)

        # Waveform plot
        self.fig, self.ax = plt.subplots(figsize=(10, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()

        # Initialize play marker (vertical line)
        self.play_marker = self.ax.axvline(x=0, color='r')

        # Mouse event
        self.canvas.mpl_connect('button_press_event', self.on_click)

        # Slider for zoom control
        zoom_frame = tk.Frame(self.master)
        zoom_frame.pack(side=tk.TOP, pady=5)
        zoom_label = tk.Label(zoom_frame, text="Zoom:")
        zoom_label.pack(side=tk.LEFT)
        self.zoom_slider = tk.Scale(zoom_frame, from_=1, to=10, orient=tk.HORIZONTAL, command=self.update_zoom)
        self.zoom_slider.pack(side=tk.LEFT)

        # Scrollbar for panning
        self.scrollbar = tk.Scrollbar(self.master, orient=tk.HORIZONTAL, command=self.update_offset)
        self.scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.scrollbar.config(command=self.update_offset)
        self.scrollbar.set(0, 0.1)

        # Update timer
        self.update_interval = 100  # milliseconds
        self.after_id = None

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if file_path:
            try:
                with wave.open(file_path, 'rb') as wf:
                    self.fs = wf.getframerate()
                    n_channels = wf.getnchannels()
                    sampwidth = wf.getsampwidth()
                    n_frames = wf.getnframes()
                    audio_frames = wf.readframes(n_frames)

                if sampwidth == 2:
                    data = np.frombuffer(audio_frames, dtype=np.int16)
                    data = data / 32768.0  # Normalize 16-bit PCM
                else:
                    messagebox.showerror("Error", f"Unsupported sample width: {sampwidth * 8} bits")
                    return

                if n_channels > 1:
                    data = data.reshape(-1, n_channels)
                    data = data.mean(axis=1)  # Convert to mono
                self.data = data
                self.duration = len(self.data) / self.fs
                self.end_idx = len(self.data)
                self.start_idx = 0
                self.plot_waveform()
                self.play_button.config(state=tk.NORMAL)
                self.stop_button.config(state=tk.NORMAL)
            except wave.Error as e:
                messagebox.showerror("Error", f"Failed to load WAV file.\n{e}")
            except Exception as e:
                messagebox.showerror("Error", f"Unexpected error occurred.\n{e}")

    def plot_waveform(self):
        self.ax.clear()
        visible_duration = self.duration / self.zoom_factor
        start_time = self.offset * (self.duration - visible_duration)
        end_time = start_time + visible_duration

        start_idx = int(start_time * self.fs)
        end_idx = int(end_time * self.fs)

        if end_idx > len(self.data):
            end_idx = len(self.data)
        self.ax.plot(np.linspace(start_time, end_time, num=(end_idx - start_idx)), self.data[start_idx:end_idx], color='b')
        self.ax.set_xlim(start_time, end_time)
        self.ax.set_xlabel("Time (seconds)")
        self.ax.set_ylabel("Amplitude")
        self.ax.set_title("Waveform")
        self.play_marker = self.ax.axvline(x=self.start_idx / self.fs, color='r')  # Reset play marker
        self.canvas.draw()

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        click_time = event.xdata
        if click_time is None:
            return
        idx = int(click_time * self.fs)
        if event.button == 1:  # Left-click for start position
            self.start_idx = max(0, min(idx, len(self.data)))
            self.play_marker.set_xdata([click_time])  # Update play marker position
            self.canvas.draw()
        elif event.button == 3:  # Right-click for end position
            self.end_idx = max(0, min(idx, len(self.data)))
        self.plot_waveform()

    def play_audio(self):
        if self.is_playing:
            return
        if self.start_idx >= self.end_idx:
            messagebox.showwarning("Warning", "Invalid playback range. Start position is greater than or equal to end position.")
            return
        self.is_playing = True
        self.stop_flag = False
        self.play_thread = threading.Thread(target=self.playback)
        self.play_thread.start()
        self.update_play_marker()

    def playback(self):
        def callback(outdata, frames, time_info, status):
            if self.stop_flag:
                raise sd.CallbackStop()
            nonlocal pos
            end = pos + frames
            if end > self.end_idx:
                end = self.end_idx
                frames = end - pos
                outdata[:frames] = self.data[pos:end].reshape(-1, 1)
                outdata[frames:] = 0
                raise sd.CallbackStop()
            else:
                outdata[:] = self.data[pos:end].reshape(-1, 1)
                pos = end

        pos = self.start_idx
        try:
            with sd.OutputStream(samplerate=self.fs, channels=1, dtype='float32', callback=callback):
                while self.is_playing and not self.stop_flag and pos < self.end_idx:
                    time.sleep(0.02)
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during playback.\n{e}")
        self.is_playing = False

    def stop_audio(self):
        if not self.is_playing:
            return
        self.stop_flag = True
        self.is_playing = False
        if self.play_thread is not None:
            self.play_thread.join()
        if self.after_id is not None:
            self.master.after_cancel(self.after_id)
            self.after_id = None

    def update_play_marker(self):
        if not self.is_playing:
            return
        current_time = self.start_idx / self.fs
        self.play_marker.set_xdata([current_time])  # Update play marker position (リストに変更)
        self.ax.figure.canvas.draw_idle()
        self.start_idx += int((self.fs * self.update_interval) / 1000)
        if self.start_idx >= self.end_idx:
            self.is_playing = False
            return
        self.after_id = self.master.after(self.update_interval, self.update_play_marker)

    def update_zoom(self, val):
        self.zoom_factor = float(val)
        self.plot_waveform()

    def update_offset(self, *args):
        if args[0] == 'moveto':
            self.offset = float(args[1])
        elif args[0] == 'scroll':
            self.offset += float(args[1]) * 0.05  # Adjust scroll speed
        self.plot_waveform()
        self.offset = float(args[1])
        self.plot_waveform()


def main():
    root = tk.Tk()
    app = AudioPlayer(root)
    root.mainloop()


if __name__ == "__main__":
    main()