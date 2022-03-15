#%%
import numpy as np
import pandas as pd
from utils import load_data_input, plot_sequence
import cv2 
from transform_output_format import get_4D_output
import matplotlib.pyplot as plt
from tqdm import tqdm
# %%
GHI,CLS,SZA,SAA,dates = load_data_input("X_train_copernicus.npz")
y_train_csv = pd.read_csv('y_train_zRvpCeO_nQsYtKN.csv')
y_train = get_4D_output(y_train_csv)

def benchmark(sequence, sequence_cls):
    last_image = sequence[-1].copy()
    preds = []
    for i in range(4):
        preds.append(last_image*(sequence_cls[4+i]/sequence_cls[3]))
    return np.array(preds)[:,15:66,15:66]
#%%
errors = []
for idx in tqdm(range(GHI.shape[0])):
    prev_img = GHI[idx,2]
    next_img = GHI[idx,3]
    flow = cv2.calcOpticalFlowFarneback(prev_img,next_img, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # plt.imshow(aa[:,:,1])

    h = flow.shape[0]
    w = flow.shape[1]
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    ratio_cls = CLS[idx,4]/CLS[idx,3]
    new_frame = (cv2.remap(next_img*ratio_cls,flow, None, cv2.INTER_LINEAR))[15:66,15:66]
    bench = benchmark(GHI[idx], CLS[idx])
    error_1 = np.sum((new_frame-y_train[idx,0])**2)
    error_2 = np.sum((bench[0]-y_train[idx,0])**2)
    errors.append(error_2-error_1)
    # print(np.sum((new_frame-y_train[idx,0])**2))
    # print(np.sum((bench[0]-y_train[idx,0])**2))
# %%
def optical_flow(sequence, sequence_cls):
    prev_img = sequence[-2]
    next_img = sequence[-1]
    preds = []
    flow = cv2.calcOpticalFlowFarneback(prev_img,next_img, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    h,w = flow.shape[:2]
    for i in range(4):
        flow_2 = (i+1)*flow
        flow_2[:,:,1] += np.arange(h)[:,np.newaxis]
        flow_2[:,:,0] += np.arange(w)
        new_frame = cv2.remap(next_img, flow_2, None, cv2.INTER_LINEAR)*(sequence_cls[4+i]/sequence_cls[3])
        preds.append(new_frame)
    return np.array(preds)[:,15:66,15:66]

# %%



#%%
idx = 862
seq = optical_flow(GHI[idx], CLS[idx])
pers = benchmark(GHI[idx], CLS[idx])
error_of = np.sum((seq-y_train[idx])**2)
error_pers = np.sum((pers-y_train[idx])**2)
print(f'{error_of} pour OF')
print(f'{error_pers} pour benchamrk')
print(error_of-error_pers)
# %%
errors = []
idx=0
for seq_ghi, seq_cls in tqdm(zip(GHI,CLS)):
    preds_of = optical_flow(seq_ghi, seq_cls)
    preds_pers = benchmark(seq_ghi, seq_cls)
    error_of = np.sum((preds_of-y_train[idx])**2)
    error_pers = np.sum((preds_pers-y_train[idx])**2)
    errors.append(error_of-error_pers)
    idx+=1
# %%
errors = np.array(errors)
plt.plot(errors)
np.count_nonzero(np.where(errors>0))
# %%
