from torch import nn


class SimpleTestNet(nn.Module):
    def __init__(self, num_units=10, nonlin=nn.ReLU()):
        super(SimpleTestNet, self).__init__()

        self.dense0 = nn.Linear(20, num_units)
        self.nonlin = nonlin
        self.dropout = nn.Dropout(0.5)
        self.dense1 = nn.Linear(num_units, num_units)
        self.output = nn.Linear(num_units, 2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        X = self.dropout(X)
        X = self.nonlin(self.dense1(X))
        X = self.softmax(self.output(X))
        return X


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

def get_lenet():
    net = torch.nn.Sequential(
        Reshape(),
        nn.Conv2d(3, 6, kernel_size=5, padding=2), nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(16 * 22 * 22, 120), nn.Sigmoid(), # TODO: add more Conv Layers before
        nn.Linear(120, 10), nn.Sigmoid(),
        nn.Linear(10, 1)) # one output as we only have one (positive) class
    return net
 