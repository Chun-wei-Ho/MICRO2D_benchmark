import torch
from torch.utils.data import Dataset

from torchvision import transforms

import numpy as np
import h5py

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
    structs_train, structs_test, Ex_train, Ex_test = structs[:-1000], structs[-1000:], Ex[:-1000], Ex[-1000:]
    return MICRO2DDataset(structs_train, Ex_train), MICRO2DDataset(structs_test, Ex_test)

class MICRO2DDataset(Dataset):
    def __init__(self, structs, Ex):
        self.structs = structs
        self.Ex = Ex

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
        ])
    def __len__(self):
        return self.Ex.shape[0]
    def __getitem__(self, idx):
        return self.transforms(self.structs[idx]), self.Ex[idx]

class DataAug(Dataset):
    def __init__(self, dataset):
        self.transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            # transforms.RandomRotation(degrees=45),
            # transforms.RandomInvert(p=0.5),
        ])
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        structs, Ex = self.dataset[idx]
        return self.transforms(structs), Ex

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    train_dataset, test_dataset = load_train_test("MICRO2D_homogenized.h5")
    train_dataloader = DataLoader(train_dataset, batch_size=32, drop_last=True, shuffle=True)
    for struct, Ex in train_dataloader:
        print(struct.shape, Ex.shape)