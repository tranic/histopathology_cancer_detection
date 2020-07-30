from torch import nn
from torchvision import models
import torch

"""
This file contains all CNN architectures as classes (non-pretrained and pretrained) inheriting from torch.nn.Module.
"""


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        self.ReLu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()

        self.conv2d_0 = nn.Conv2d(3, 6, kernel_size=5, padding=2)
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2d_2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear_4 = nn.Linear(16 * 22 * 22, 120)
        self.linear_5 = nn.Linear(120, 10)
        # two outputs for softmax in final layer
        self.output = nn.Linear(10, 1)

    def forward(self, X, **kwargs):
        X = X.view(-1, 3, 96, 96).float()

        X = self.ReLu(self.conv2d_0(X))
        X = self.pool_1(X)
        X = self.ReLu(self.conv2d_2(X))
        X = self.pool_3(X)
        X = self.flatten(X)
        X = self.ReLu(self.linear_4(X))
        X = self.ReLu(self.linear_5(X))

        X = self.output(X)

        return X


class DenseNet121(nn.Module):
    def __init__(self):
        super(DenseNet121, self).__init__()

        base_net = models.densenet121(pretrained=False)

        self.features = base_net.features

        self.dense121_relu = nn.ReLU(inplace=True)
        self.dense121_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(nn.Linear(1024, 512),
                                        nn.Dropout(p=0.1),
                                        nn.ReLU(),
                                        nn.Linear(512, 1))

        del base_net

    def forward(self, X):
        X = X.view(-1, 3, 96, 96).float()

        X = self.features(X)

        # Convert output of predifined dense121 layers to a format that can used by the classifier "layers"
        X = self.dense121_relu(X)
        X = self.dense121_pool(X)
        X = torch.flatten(X, 1)

        X = self.classifier(X)

        return X


class DenseNet121Pretrained(nn.Module):
    def __init__(self):
        super(DenseNet121Pretrained, self).__init__()

        base_net = models.densenet121(pretrained=True)

        self.features = base_net.features

        self.dense121_relu = nn.ReLU(inplace=True)
        self.dense121_pool = nn.AdaptiveAvgPool2d((1, 1))

        # we only want to train the last layers
        for param in list(self.features.parameters()):
            param.requires_grad = False  # as default is True for all
        for param in list(self.features.denseblock4.denselayer13.parameters()):
            param.requires_grad = True
        for param in list(self.features.denseblock4.denselayer14.parameters()):
            param.requires_grad = True
        for param in list(self.features.denseblock4.denselayer15.parameters()):
            param.requires_grad = True
        for param in list(self.features.denseblock4.denselayer16.parameters()):
            param.requires_grad = True
            
            
        self.classifier = nn.Sequential(nn.Linear(1024, 512),
                                        nn.Dropout(p=0.1),
                                        nn.ReLU(),
                                        nn.Linear(512, 1))

        del base_net

    def forward(self, X):
        X = X.view(-1, 3, 224, 224).float()

        X = self.features(X)
        
        X = self.dense121_relu(X)
        X = self.dense121_pool(X)
        X = torch.flatten(X, 1)

        X = self.classifier(X)

        return X


class DenseNet201(nn.Module):
    def __init__(self):
        super(DenseNet201, self).__init__()

        base_net = models.densenet201(pretrained=False)

        self.features = base_net.features

        self.dense201_relu = nn.ReLU(inplace=True)
        self.dense201_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(nn.Linear(1920, 512),
                                        nn.Dropout(p=0.1),
                                        nn.ReLU(),
                                        nn.Linear(512, 1))

        del base_net

    def forward(self, X):
        X = X.view(-1, 3, 96, 96).float()

        X = self.features(X)

        X = self.dense201_relu(X)
        X = self.dense201_pool(X)
        X = torch.flatten(X, 1)

        X = self.classifier(X)

        return X


class DenseNet201Pretrained(nn.Module):
    def __init__(self):
        super(DenseNet201Pretrained, self).__init__()

        base_net = models.densenet201(pretrained=True)

        self.features = base_net.features

        self.dense201_relu = nn.ReLU(inplace=True)
        self.dense201_pool = nn.AdaptiveAvgPool2d((1, 1))

        # we only want to train the last layers
        for param in list(self.features.parameters()):
            param.requires_grad = False  # as default is True for all
        for param in list(self.features.denseblock4.denselayer29.parameters()):
            param.requires_grad = True
        for param in list(self.features.denseblock4.denselayer30.parameters()):
            param.requires_grad = True
        for param in list(self.features.denseblock4.denselayer31.parameters()):
            param.requires_grad = True
        for param in list(self.features.denseblock4.denselayer32.parameters()):
            param.requires_grad = True

        self.classifier = nn.Sequential(nn.Linear(1920, 512),
                                        nn.Dropout(p=0.1),
                                        nn.ReLU(),
                                        nn.Linear(512, 1))

        del base_net

    def forward(self, X):
        X = X.view(-1, 3, 224, 224).float()

        X = self.features(X)

        X = self.dense201_relu(X)
        X = self.dense201_pool(X)
        X = torch.flatten(X, 1)

        X = self.classifier(X)

        return X


class ResNet18_96(nn.Module):
    def __init__(self):
        super(ResNet18_96, self).__init__()

        self.model = models.resnet18(pretrained=False)

        # change last layer (fc) to adjust for binary classification
        n_features_in = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(n_features_in, 1)
        )

    def forward(self, X):
        X = X.view(-1, 3, 96, 96).float()

        X = self.model(X)
        return X


class ResNet152_96(nn.Module):
    def __init__(self, pretrained=False):
        super(ResNet152_96, self).__init__()

        self.model = models.resnet152(pretrained=pretrained)
        print("pretrained=", pretrained)

        if pretrained:
            # we only want to train the last 2 multilayers (i.e., layer 3 and 4)
            for param in list(self.model.parameters()):
                param.requires_grad = False  # as default is True for all
            for param in list(self.model.layer3.parameters()):
                param.requires_grad = True
            for param in list(self.model.layer4.parameters()):
                param.requires_grad = True

        # change last layer (fc) to adjust for binary classification
        n_features_in = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(n_features_in, 1)
        )

    def forward(self, X):
        X = X.view(-1, 3, 96, 96).float()

        X = self.model(X)
        return X


class ResNet34Pretrained(nn.Module):
    def __init__(self):
        super(ResNet34Pretrained, self).__init__()

        self.model = models.resnet34(pretrained=True)

        # we only want to train the last 2 multilayers (i.e., layer 3 and 4)
        for param in list(self.model.parameters()):
            param.requires_grad = False  # as default is True for all
        for param in list(self.model.layer3.parameters()):
            param.requires_grad = True
        for param in list(self.model.layer4.parameters()):
            param.requires_grad = True

        # change last layer (fc) to adjust for binary classification
        n_features_in = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(n_features_in, 1)
        )

    def forward(self, X):
        X = X.view(-1, 3, 224, 224).float()

        X = self.model(X)
        return X


class ResNet152Pretrained(nn.Module):
    def __init__(self):
        super(ResNet152Pretrained, self).__init__()

        self.model = models.resnet152(pretrained=True)

        # we only want to train the last 2 multilayers (i.e., layer 3 and 4)
        for param in list(self.model.parameters()):
            param.requires_grad = False  # as default is True for all
        for param in list(self.model.layer3.parameters()):
            param.requires_grad = True
        for param in list(self.model.layer4.parameters()):
            param.requires_grad = True

        # change last layer (fc) to adjust for binary classification
        n_features_in = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(n_features_in, 1)
        )

    def forward(self, X):
        X = X.view(-1, 3, 224, 224).float()

        X = self.model(X)
        return X


class VGG11(nn.Module):
    def __init__(self):
        super(VGG11, self).__init__()

        base_net = models.vgg11(pretrained=False)

        self.features = base_net.features


        self.avgpool = base_net.avgpool

        self.classifier = nn.Sequential(

            nn.Linear(in_features=25088, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=1, bias=True)
        )

        del base_net

    def forward(self, X):
        X = X.view(-1, 3, 96, 96).float()

        X = self.features(X)
        X = self.avgpool(X)
        X = torch.flatten(X, 1)
        X = self.classifier(X)

        return X


class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()

        base_net = models.vgg19(pretrained=False)

        self.features = base_net.features

        self.avgpool = base_net.avgpool

        self.classifier = nn.Sequential(

            nn.Linear(in_features=25088, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=1, bias=True)
        )

        del base_net

    def forward(self, X):
        X = X.view(-1, 3, 96, 96).float()

        X = self.features(X)
        X = self.avgpool(X)
        X = torch.flatten(X, 1)
        X = self.classifier(X)

        return X
