import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from d2l import torch as d2l

# load a small data set
def load_data(path, batchsize, num_workers, ids=-1):
    image_ids = ids
    y = []  # labels

    dataset = torchvision.datasets.ImageFolder(
        root = path,
        transform = torchvision.transforms.ToTensor()
    )

    loader = DataLoader(dataset=dataset, batch_size=batchsize, num_workers=num_workers, shuffle=True)
    return loader

"""
class Reshape(torch.nn.Module):
    def forward(self, x):
        return x.view(-1,1,96,96)"""

def get_lenet():
    net = torch.nn.Sequential(
        # Reshape(),
        nn.Conv2d(3, 6, kernel_size=5, padding=2), nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(16 * 22 * 22, 120), nn.Sigmoid(), # TODO: add more Conv Layers before
        nn.Linear(120, 10), nn.Sigmoid(),
        nn.Linear(10, 1)) # one output as we only have one (positive) class
    return net

def train_model(net, train_iter, test_iter, num_epochs, lr,
              device=d2l.try_gpu()):
    # for idx, (X, y) in enumerate(train_iter):

    """Train and evaluate a model with CPU or GPU."""
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            torch.nn.init.xavier_uniform_(m.weight)  # Part 2.2

    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', xlim=[0, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer = d2l.Timer()
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)  # train_loss, train_acc, num_examples
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            net.train()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            output = net(X)
            y_hat = torch.round(torch.exp(output)/(1+torch.exp(output)))
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_loss, train_acc = metric[0] / metric[2], metric[1] / metric[2]
            if (i + 1) % 50 == 0:
                animator.add(epoch + i / len(train_iter),
                             (train_loss, train_acc, None))
        # test_acc = evaluate_accuracy_gpu(net, test_iter) TODO: add test data loader
        # animator.add(epoch + 1, (None, None, test_acc))
    print('loss %.3f, train acc %.3f' % ( # , test acc %.3f
        train_loss, train_acc))# , test_acc
    print('%.1f examples/sec on %s' % (
        metric[2] * num_epochs / timer.sum(), device))



def evaluate_accuracy_gpu(net, data_iter, device=None): #@save
    if not device:
        device = next(iter(net.parameters())).device
    metric = d2l.Accumulator(2)  # num_corrected_examples, num_examples
    for X, y in data_iter:
        X, y = X.to(device), y.to(device)
        metric.add(d2l.accuracy(net(X), y), sum(y.shape))
    return metric[0] / metric[1]

if __name__ == '__main__':
    ### params ---------------------------------------------------------------------------------------------------------
    data_path = "data/train/"
    batchsize = 128
    image_ids = []
    load_num_workers = 0
    ### ----------------------------------------------------------------------------------------------------------------

    # TODO: maybe crop images before (e.g., to 48x48)

    train_iter = load_data(path="data/subset_train", batchsize=batchsize, num_workers=load_num_workers)
    # TODO: debug "loss 0.000, train acc 128.000"
    # test_iter =
    net = get_lenet()

    ## TODO: explore data (num images, class distribution)

    train_model(net, train_iter, "placeholder", 20, 0.9)