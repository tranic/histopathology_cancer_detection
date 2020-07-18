from torch import nn
from torchvision import models


class LeNet(nn.Module):
    def __init__(self, nonlin = nn.Sigmoid()):
        super(LeNet, self).__init__()
        
        self.nonlin = nonlin
        self.flatten = nn.Flatten()
        self.softmax = nn.Softmax(dim = -1)

        self.conv2d_0 = nn.Conv2d(3, 6, kernel_size = 5, padding = 2)
        self.pool_1 =  nn.AvgPool2d(kernel_size = 2, stride = 2)
        self.conv2d_2 = nn.Conv2d(6, 16, kernel_size = 5)
        self.pool_3 =  nn.AvgPool2d(kernel_size = 2, stride = 2)
        self.linear_4 = nn.Linear(16 * 22 * 22, 120) # TODO: add more Conv Layers before
        self.linear_5 = nn.Linear(120, 10)
        self.output = nn.Linear(10, 2) # two outputs for softmax in final layer
        
        

    def forward(self, X, **kwargs):
        X = X.view(-1, 3, 96, 96).float() # previously Reshape()
        
        X = self.nonlin(self.conv2d_0(X))
        X = self.pool_1(X)
        X = self.nonlin(self.conv2d_2(X))
        X = self.pool_3(X)
        X = self.flatten(X)
        X = self.nonlin(self.linear_4(X))
        X = self.nonlin(self.linear_5(X))
        
        X = self.softmax(self.output(X))
        
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
 
    
