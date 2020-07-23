from torch import nn
from torchvision import models
import torch

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        
        self.ReLu = nn.nn.ReLU(inplace = True)
        self.flatten = nn.Flatten()

        self.conv2d_0 = nn.Conv2d(3, 6, kernel_size = 5, padding = 2)
        self.pool_1 =  nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv2d_2 = nn.Conv2d(6, 16, kernel_size = 5)
        self.pool_3 =  nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.linear_4 = nn.Linear(16 * 22 * 22, 120) # TODO: add more Conv Layers before
        self.linear_5 = nn.Linear(120, 10)
        self.output = nn.Linear(10, 2) # two outputs for softmax in final layer
        
        

    def forward(self, X, **kwargs):
        X = X.view(-1, 3, 96, 96).float() # previously Reshape()
        
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
        
        # Load dene121 net 
        base_net = models.densenet121(pretrained = False)
        
        self.sigmoid = nn.Sigmoid()
        
        # Exctract all dense121 layers for own use
        self.features = base_net.features
        
        # Change input layer of dense121 to match our input sizeß
        self.features.conv0 = nn.Conv2d(3, 64, kernel_size = 3, stride = 2, padding = 1, bias = False)
        self.features.norm0 = nn.BatchNorm2d(64)
        self.features.relu0 = nn.ReLU(inplace = True)
        
        self.dense121_relu = nn.ReLU(inplace = True)
        self.dense121_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        
        # maybe rename to something else, only c3 is the actually classifier
        self.classifier0 = nn.Linear(1024, 512) 
        self.classifier1 = nn.Dropout(p = 0.1)
        self.classifier2 = nn.ReLU()
        self.classifier3 = nn.Linear(512, 1)
       
        del base_net
        
    def forward(self, X):
        X = X.view(-1, 3, 96, 96).float()
        
        X = self.features(X)
    
        # Convert output of predifined dense121 layers to a format that can used by the classifier "layers"
        X = self.dense121_relu(X)
        X = self.dense121_pool(X)
        X = X.view(X.size(0), -1)
        
        X = self.classifier0(X)
        X = self.classifier1(X)
        X = self.classifier2(X)
        X = self.classifier3(X)
        
        return X


class ResNet34(nn.Module):
    def __init__(self):
        super(ResNet34, self).__init__()

        # Load resnet34
        self.model = models.resnet34(pretrained=True)

        # we only want to train the last 2 multilayers (i.e., layer 3 and 4)
        for param in list(self.model.parameters()):
            param.requires_grad = False # as default is True for all
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


class ResNet18(nn.Module):
    def __init__(self, pretrained=False):
        super(ResNet18, self).__init__()

        self.model = models.resnet18(pretrained=pretrained)

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
        X = X.view(-1, 3, 224, 224).float()

        X = self.model(X)
        return X

class ResNet152(nn.Module):
    def __init__(self, pretrained=False):
        super(ResNet152, self).__init__()

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
        X = X.view(-1, 3, 224, 224).float()

        X = self.model(X)
        return X
 
class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        
        # Load dene121 net 
        base_net = models.vgg16(pretrained = False)


        # Exctract all dense121 layers for own use
        self.features = base_net.features
        
        # Change input layer of dense121 to match our input sizeß
        self.features[0] = nn.Conv2d(3, 64, kernel_size = 3, stride = 2, padding = 1, bias = False)

        self.avgpool = base_net.avgpool

        self.classifier2 = nn.Sequential(
            
            nn.Linear(in_features=25088, out_features=4096, bias=True),
            nn.ReLU(inplace = True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace = True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=1, bias=True)
            )

       
        del base_net
        
    def forward(self, X):
        X = X.view(-1, 3, 96, 96).float()
        
        X = self.features(X)
        X = self.avgpool(X)
        X = torch.flatten(X, 1)
        X = self.classifier2(X)

        
        return X
    
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        
        self.layer = nn.Sequential(models.resnet18(pretrained=False, num_classes=1000),nn.Linear(1000,512), nn.ReLU(inplace=True), nn.Linear(512,1),nn.Sigmoid())

    def forward(self, X):
        
        X = X.view(-1, 3, 96, 96).float()
        
        X = self.layer(X)
      
        return X

