# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from transform_output_format import get_4D_output, get_2D_output
import datetime
# %%
data = np.load('X_train_copernicus.npz', allow_pickle=True)
dates = data['datetime']
GHI = data['GHI']
CLS = data['CLS']
SZA = data['SZA']
SAA = data['SAA']

# %%
idx = 99

def plot_sequence(sequence=None, index_to_plot=0):
    if sequence is None:
        sequence = GHI[index_to_plot]
    fig, ax = plt.subplots(2,2, figsize=(8,8))
    max_v = np.max(sequence)
    for i in range(2):
        for j in range(2):
            im = ax[i,j].imshow(sequence[j+2*i], cmap='jet', vmin=0, vmax=max_v)
            fig.colorbar(im, ax=ax[i,j])
            ax[i,j].set_title(str(dates[index_to_plot]+datetime.timedelta(minutes=15*(j+2*i))))

    plt.show()

plot_sequence(GHI[1242])

# %%
y_train_csv = pd.read_csv('y_train_zRvpCeO_nQsYtKN.csv')
y_train = get_4D_output(y_train_csv)
# %%
idx = 800
fig, ax = plt.subplots(2,4, figsize=(10,7))
for i in range(4):
    ax[0,i].imshow(GHI[idx,i,15:66,15:66], cmap='jet')
    ax[1,i].imshow(y_train[idx,i], cmap='jet')
    ax[0,i].set_title(str(dates[idx]+datetime.timedelta(minutes=15*i)))
    ax[1,i].set_title(str(dates[idx]+datetime.timedelta(minutes=15*(4+i))))
# %%

def benchmark(sequence, sequence_cls):
    last_image = sequence[-1].copy()
    preds = []
    for i in range(4):
        preds.append(last_image*(sequence_cls[4+i]/sequence_cls[3]))
    return np.array(preds)[:,15:66,15:66]

# %%
idx = 875
sequence = benchmark(GHI[875], CLS[875])
plot_sequence(sequence)
ground_truth = y_train[idx]
plot_sequence(ground_truth)
# %%
data_test = np.load("X_test_copernicus.npz", allow_pickle=True)
GHI_test = data_test['GHI']
CLS_test = data_test['CLS']

# %%
preds_benchmark = []

for ghi, cls in zip(GHI_test, CLS_test):
    preds_benchmark.append(benchmark(ghi, cls))

preds_benchmark = np.array(preds_benchmark)
# %%
aa = get_2D_output(preds_benchmark)
# %%
aa.to_csv('test.csv', index=False)
# %%
