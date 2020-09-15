#!/usr/bin/env python
# coding: utf-8


import os

import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import sys

np.seterr(divide='ignore', invalid='ignore')

class app():
    """
    The app takes audio input and generates spectrograms \
    with option of spectral centroid overlay.
    """
    def __init__(self):
        """Initiate with configurations switch statement."""
        arguments = input('Would you like to plot the spectral centroid? [Y/n]\n')
        run = self.config(arguments)
        run()
    
    def config(self, arguments):
        config = {
            'Y': self.spectral_centroid,
            'n': self.spectrogram
        }
        
        return config.get(arguments, "No specified configurations. Only spectrogram will be annotated.")
    
    def load_files(self):
        """Finds and loads files from within folder structure."""
        input_path = './input'
        
        #Ignores system files on mac
        files = [x for x in os.listdir(input_path) if x != '.DS_Store' and x != 'README.md']
        
        return files
    

    def moving_average(self, a, n=30) :
        """Creates a moving average to apply smoothing to the spectral centroid plot."""
        self.window = n
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        
        return ret[n - 1:] / n

    def spectral_centroid(self):
        """Plot the spectral centroid overlay."""
        files = self.load_files()
        save_path = './output/'
        for file in files:
            print('Working on file: ' + file)
            fname = file.split('.',2)[0]
            y, sr = librosa.load('./input/' + file, sr=48000)
            cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            cent = self.moving_average(cent)
            S, phase = librosa.magphase(librosa.stft(y=y))
            freqs, times, D = librosa.reassigned_spectrogram(y, fill_nan=True)
            times = librosa.times_like(cent)
            fig, ax = plt.subplots(figsize=(30,20))
            librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                                     y_axis='log', x_axis='time', ax=ax)
            ax.plot(times, cent.T, label='Spectral centroid', color='w')
            ax.legend(loc='upper right')
            ax.set(title=f'log Power spectrogram {fname} \n Moving Average: {self.window}')
            
            
            plt.savefig(''.join([save_path, fname, '_spectral_centroid.png']), format='png')
            
    
    def spectrogram(self):
        """Plots a spectrogram."""
        files = self.load_files()
        save_path = './output/'
        for file in files:
            print('Working on file: ' + file)
            fname = file.split('.',2)[0]
            y, sr = librosa.load('./input/' + file, sr=48000)
            fig, ax = plt.subplots(figsize=(30,20))
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,
                                    fmax=8000)
            S_dB = librosa.power_to_db(S, ref=np.max)
            img = librosa.display.specshow(S_dB, x_axis='time',
                                     y_axis='mel', sr=sr,
                                     fmax=8000, ax=ax)
            fig.colorbar(img, ax=ax, format='%+2.0f dB')
            ax.set(title=f'Mel-frequency spectrogram {fname}')
            
            plt.savefig(''.join([save_path, fname, '_spectrogram.png']), format='png')
        
if __name__ == "__main__":
    # if sys.version_info < (3, 0):
    #     # Python 2
    #     import Tkinter as tk

    # else:
    #     # Python 3
    #     import tkinter as tk
    # root = tk.Tk()
    # root.title("Music Information Retrieval Analysis Tool.")
    # lbl = tk.Label(root, text="Include spectral centroid?")
    # lbl.grid(column=0, row=0)
    # lbl.pack()
    # yes = tk.Button(root, text="Yes", command=lambda: app('Y'))
    # yes.pack()
    # no = tk.Button(root, text="No", command=lambda: app('n'))
    # no.pack()
    # tk.mainloop()
    app()

