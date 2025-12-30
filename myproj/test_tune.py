import h5py
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import scipy.signal
import heapq
import time


# Weighted interpolation
def findpeaks(x, K, DOA):
    x = np.array(x)
    indexes, _ = scipy.signal.find_peaks(x)
    maxi = heapq.nlargest(K, x[indexes])
    ind = np.zeros((K,))
    ind = np.int_(ind)
    p = np.zeros((K,))

    if len(indexes) == 0:
        ind[0] = 60
        ind[1] = 59
        ind = np.int_(ind)
        p = DOA[ind]
    else:
        if len(indexes) < K:
            ind[0] = indexes
            ind[1] = 60
            ind = np.int_(ind)
            p = DOA[ind]
        else:
            for i in range(K):
                ind[i] = np.where(x == maxi[i])[0][0]
                ind[i] = np.int_(ind[i])

                if ind[i] == 0:
                    p[i] = DOA[ind[i]]
                else:
                    l = int(ind[i] - 1)
                    r = int(ind[i] + 1)
                    ind[i] = np.int_(ind[i])

                    if x[l] < x[r]:
                        p[i] = x[r] / (x[r] + x[ind[i]]) * DOA[r] + x[ind[i]] / (x[r] + x[ind[i]]) * DOA[ind[i]]
                    else:
                        p[i] = x[l] / (x[l] + x[ind[i]]) * DOA[l] + x[ind[i]] / (x[l] + x[ind[i]]) * DOA[ind[i]]

    ind = np.int_(ind)

    return p, DOA[ind]

device = torch.device("cpu")
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Select model (Change to your own path)
model = torch.load(r'/home/abc/PycharmProjects/myProjects/models/Fine_tuning_ablation_smallDGCNet/model_10.pth',
                   map_location=device, weights_only=False).to(device)

# Test Data Path (Change to your own path)
f_data_root = '/home/abc/PycharmProjects/myProjects/data/'
f_result_root = '/home/abc/PycharmProjects/myProjects/Results/test_data/'
f_data = f_data_root + 'DOA_test_K_7.h5'
f_result = f_result_root + 'EX1_result_DOA_test_K_1_-20DB.h5'

# Load the test data
data_file = h5py.File(f_data, 'r')
GT_angles = np.transpose(np.array(data_file['angle']))  # Ground Truth angles
Ry_sam_test = np.array(data_file['SCM'])
test_data = Ry_sam_test
test_data = torch.tensor(test_data, dtype=torch.float32).unsqueeze(0).to(device)

# Scenario prior parameters
source_num = len(GT_angles[0])
res = 1
An_max = 60
An_min = -60
v = np.arange(An_min, An_max + res, res)

# DOA estimation obtained through model reconstruction of spatial spectrum
start_time = time.time()
model.eval()
y_pred_sam = model(test_data).detach().cpu().numpy()

# Original method: take the maximum position directly
source_index = np.argpartition(y_pred_sam, -source_num, axis=-1)[:, -source_num:].reshape(-1)
DOA_esti_raw = v[source_index]

# Interpolation method: findpeaks function is used for peak refinement
DOA_esti_refined, DOA_esti_peaks = findpeaks(y_pred_sam[0, :], source_num, v)
end_time = time.time()
print(f"Run time: {end_time - start_time:.6f} s")

print('Ground Truth angles:', GT_angles[0])
print('Original DOA estimation results: ', DOA_esti_raw)
print('Refined DOA estimation results: ', DOA_esti_refined)

# Drawing spatial spectrum
plt.figure(figsize=(8, 6))
plt_GT = plt.plot(GT_angles, 1, "or", label = 'Ture DOA')
plt_Esti = plt.plot(DOA_esti_refined, np.ones(len(DOA_esti_refined)), "*b", label='Esti DOA')
plt.legend(handles=[plt_GT[0], plt_Esti[0]], labels=['Ture DOA', 'Esti DOA'])
plt.plot(v, y_pred_sam[0, :])
plt.text(-60, 1.15, "GT:")
for i in GT_angles[0]:
        plt.text(i, 1.15, i)

plt.text(-60, 1.1, "esti:")
for i in DOA_esti_refined:
        plt.text(i,1.1, round(i, 1))
plt.xlabel('Angle')
plt.ylabel('Spectrum')
plt.grid()
plt.savefig('hybridvit_k=7.svg', dpi=300, bbox_inches='tight') # (Change to your own path)
plt.show()

# Save the reconstructed spatial spectrum
if os.path.exists(f_result_root) is False:
        os.makedirs(f_result_root)
hf = h5py.File(f_result, 'w')
hf.create_dataset('GT_angles', data=GT_angles.T)
hf.create_dataset('spectrum', data=y_pred_sam)