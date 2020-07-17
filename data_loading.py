import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import os
import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np

class HistopathDataset(Dataset):
    """ Histopathologic Cancer Dataset that represents a map from keys to data samples."""

    def __init__(self, label_file, root_dir, transform=None, greyscale=False):
        """
        :param label_file: path to csv file containing labels for each and every image id of the data set.
        :param root_dir:  path to directory with all images.
        :param transform: optional transformation operation on each sample.
        """
        self.data_frame = pd.read_csv(label_file)
        self.root_dir = root_dir
        self.transform = transform
        self.greyscale = greyscale

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, index):
        """
         Fetch a data sample for a given key and turn it to greyscale.
        """
        if torch.is_tensor(index):
            index = index.tolist()

        img_path = os.path.join(self.root_dir, self.data_frame.iloc[index, 0])
        img_path = img_path + ".tif"
        image = io.imread(fname=img_path, as_gray=self.greyscale) # np ndarray
        label = self.data_frame.iloc[index, 1]
        sample = (image, label)

        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor(object):
    """Convert ndarray from sample to Tensor."""

    def __call__(self, sample):
        image, label = sample[0], sample[1]

        # numpy image: H x W x Cß
        # torch image: C X H X W
        image = image.transpose((2, 0, 1)) # for colored images
        return (torch.from_numpy(image), label.astype(np.float32))

if __name__ == '__main__':
    ## Example on how to use the HistopathDataset class
    num_workers = 0
    batchsize = 128

    # create custom dataset
    transformed_dataset = HistopathDataset(
        label_file=os.path.abspath("data/train_labels.csv"),
        root_dir=os.path.abspath("data/train"),
        transform=ToTensor())

    # get images manually: plot first two images
    for i in range(len(transformed_dataset)):
        sample = transformed_dataset[i]
        print("index: ", i, " image size: ", sample[0].size(), " label: ", sample[1])
        # imgplot = plt.imshow(sample[0])
        # plt.show()
        if i == 1: break

    # use DataLoader of torch
    dataloader = DataLoader(transformed_dataset, batch_size=batchsize, shuffle=True, num_workers=num_workers)
