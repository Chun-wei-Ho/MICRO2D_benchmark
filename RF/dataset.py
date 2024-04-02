import torch
from torch.utils.data import Dataset

import numpy as np
import h5py

from sklearn.decomposition import PCA

# 2PS Cross correlations
def twopoint(phase1, phase2):
    cross_correlations = np.real(np.fft.ifftn(phase1*np.conj(phase2)) / (phase1.shape[0]*phase2.shape[0]))
    return cross_correlations

def preprocess(structs):
    TPS = []
    TPS_flat = []

    # phase 1 is the white phase, phase 2 = black
    for i in range (0,10000):
        phase1 = np.fft.fftn(structs[i]==0)
        phase2 = np.fft.fftn(structs[i]==1)
        # centering the 2 point correlations, this is where we compute the autocorrelations
        twopoint_centered = np.fft.fftshift(twopoint(phase2, phase2))
        TPS.append(twopoint_centered)
    TPS_flat = np.vstack([arr.flatten() for arr in TPS])
    # flattened TPS vectors are stacked on top of eachother. Take the mean of each column
    TPS_mean_subtracted = TPS_flat - np.mean(TPS_flat, axis=0)

    #PCA
    pca = PCA(n_components=5)
    return pca.fit_transform(TPS_mean_subtracted)

def load_train_test(filename):
    print(f"Loading data from {filename}")
    h5_file = h5py.File(filename,'r')
    structs = np.array(h5_file["GRF/GRF"], dtype=np.float32)

    homogenized_mech = np.array(h5_file["GRF/homogenized_mechanical"], dtype=np.float32)
    Props = homogenized_mech[:, 0]
    Ex = Props[:, 0]

    np.random.seed(2024)
    permutation = np.random.permutation(Ex.shape[0])
    structs, Ex = structs[permutation], Ex[permutation]

    print("Pre-processing with TPS and PCA")
    structs = preprocess(structs)

    structs_train, structs_test, Ex_train, Ex_test = structs[:-1000], structs[-1000:], Ex[:-1000], Ex[-1000:]

    return MICRO2DDataset(structs_train, Ex_train), MICRO2DDataset(structs_test, Ex_test)

class MICRO2DDataset(Dataset):
    def __init__(self, structs, Ex):
        self.structs = structs
        self.Ex = Ex
    def __len__(self):
        return self.Ex.shape[0]
    def __getitem__(self, idx):
        return self.structs[idx], self.Ex[idx]

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    train_dataset, test_dataset = load_train_test("MICRO2D_homogenized.h5")
    train_dataloader = DataLoader(train_dataset, batch_size=32, drop_last=True, shuffle=True)
    for struct, Ex in train_dataloader:
        print(struct.shape, Ex.shape)