import torch

import torchvision

class ResidualCNNblock(torch.nn.Module):
    def __init__(self, nfeatures, kernel=(3, 3)):
        super().__init__()
        self.cnn1 = torch.nn.Conv2d(nfeatures, nfeatures, kernel, padding_mode='reflect', padding='same')
        self.cnn2 = torch.nn.Conv2d(nfeatures, nfeatures, kernel, padding_mode='reflect', padding='same')
        self.batchnorm = torch.nn.BatchNorm2d(nfeatures)
    def forward(self, x):
        x1 = torch.nn.functional.relu(self.cnn1(x))
        x1 = torch.nn.functional.relu(self.cnn2(x1))
        x1 = self.batchnorm(x1)
        return x1 + x

class BasicCNN(torch.nn.Module):
    def __init__(self, in_channel=1):
        super().__init__()
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, 32, (3, 3), padding='valid'),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.25),
            ResidualCNNblock(32),
            torch.nn.Conv2d(32, 64, (6, 6), stride=3, padding='valid'),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.25),
            ResidualCNNblock(64),
            torch.nn.Conv2d(64, 128, (6, 6), stride=3, padding='valid'),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.25),
            ResidualCNNblock(128),
            torch.nn.Conv2d(128, 128, (6, 6), stride=3, padding='valid'),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.25),
        )
        self.mlp = torch.nn.Linear(6272, 1)

    def forward(self, x):
        if x.ndim == 3: x = x[:, None, ...]
        x = self.cnn(x)
        x = x.view(x.shape[0], -1)
        return self.mlp(x)[:, 0]

class Identity(torch.nn.Module):
    def forward(self, x): return x

class ResNet50(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # weights = torchvision.models.ResNeXt101_64X4D_Weights.DEFAULT
        weights = torchvision.models.ResNet50_Weights.DEFAULT
        # self.resnet = torchvision.models.resnext101_64x4d(weights=None)
        self.resnet = torchvision.models.resnet50(weights=None)
        self.preprocess = weights.transforms()
        self.resnet.fc = torch.nn.Linear(2048, 5)
    def forward(self, x):
        x = torch.cat([x, x, x], dim=1)
        x = self.preprocess(x)
        return self.resnet(x)

class DenseNet201(torch.nn.Module):
    def __init__(self):
        super().__init__()
        weights = torchvision.models.DenseNet201_Weights.DEFAULT
        self.densenet = torchvision.models.densenet201(weights=None)
        self.preprocess = weights.transforms()
        self.densenet.classifier = torch.nn.Linear(1920, 1)
    def forward(self, x):
        x = torch.cat([x, x, x], dim=1)
        x = self.preprocess(x)
        return self.densenet(x)[:, 0]

if __name__ == "__main__":
    model = BasicCNN()

    x = torch.randn((10, 256, 256))
    print(model(x).shape)