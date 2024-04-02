import torch
from torch.utils.data import Dataset

import numpy as np
import h5py

from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
import h5py
import numpy as np
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import time

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
    pca = PCA(n_components=100)
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
    import time
    start=time.time()
    train_dataset, test_dataset = load_train_test("MICRO2D_homogenized.h5")
    print(time.time()-start)
    # train_dataloader = DataLoader(train_dataset, batch_size=32, drop_last=True, shuffle=True)
    # for struct, Ex in train_dataloader:
    #     print(struct.shape, Ex.shape)
    
    models=[LinearSVR(epsilon=0.0, tol=0.0001, C=1, loss='epsilon_insensitive', random_state=2024),
        LinearSVR(epsilon=0.0, tol=0.0001, C=10, loss='epsilon_insensitive', random_state=2024),
        LinearSVR(epsilon=0.0, tol=0.0001, C=100, loss='epsilon_insensitive', random_state=2024),
        SVR(kernel="linear", C=1, gamma="auto"),
        SVR(kernel="linear", C=10, gamma="auto"),
        SVR(kernel="linear", C=100, gamma="auto"),
        SVR(kernel="poly", C=1, gamma="auto", degree=3, epsilon=0.1, coef0=1),
        SVR(kernel="poly", C=100, gamma="auto", degree=3, epsilon=0.1, coef0=1),
        SVR(kernel="rbf", C=1, gamma=0.1, epsilon=0.1),
        SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
        ]

    for i in range(len(models)):
        print("model",i)

        count=1
        svr = models[i]
        train_dataloader = DataLoader(train_dataset, batch_size=1024, drop_last=True, shuffle=True)
        for struct, Ex in train_dataloader:
            start = time.time()
            # print(count)
            # Process each batch here
            struct = struct.numpy()
            Ex = Ex.numpy()

            X_train = struct.reshape(struct.shape[0], -1)
            y_train = Ex

            svr.fit(X_train, y_train)
            count += 1
            # print(time.time()-start,"seconds")


        test_dataloader = DataLoader(test_dataset, batch_size=1024, drop_last=False, shuffle=False)

        for struct, Ex in test_dataloader:
            # Process each batch here
            start = time.time()
            struct = struct.numpy()
            Ex = Ex.numpy()

            X_test = struct.reshape(struct.shape[0], -1)

            y_test = Ex
            # print(X_test.shape, y_test.shape)
            pred_test = svr.predict(X_test)
            mse_test = mean_squared_error(y_test, pred_test)
            print("Mean Squared Error:", mse_test)
            mape_test = mean_absolute_percentage_error(y_test, pred_test)
            print("Mean Absolute Percentage Error:", mape_test)
            mae = mean_absolute_error(y_test, pred_test)
            print("Mean Absolute Error:", mae)
            # print(time.time()-start,"seconds")
    print("done")