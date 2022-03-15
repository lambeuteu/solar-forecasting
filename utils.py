import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data_input(filename):
    data = np.load(filename, allow_pickle=True)
    dates = data['datetime']
    GHI = data['GHI']
    CLS = data['CLS']
    SZA = data['SZA']
    SAA = data['SAA']
    return GHI, CLS, SZA, SAA, dates

def plot_sequence(sequence):
    fig, ax = plt.subplots(2,2, figsize=(8,8))
    max_v = np.max(sequence)
    for i in range(2):
        for j in range(2):
            im = ax[i,j].imshow(sequence[j+2*i], cmap='jet', vmin=0, vmax=max_v)
            fig.colorbar(im, ax=ax[i,j])
    plt.show()