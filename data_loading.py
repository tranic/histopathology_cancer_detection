import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import skimage.io as io
import numpy as np

from scripts.visual_helpers import Visualizer

class HistopathDataset(Dataset):
    """ Histopathologic Cancer Dataset that represents a map from keys to data samples."""

    def __init__(self, label_file, root_dir, transform=None, greyscale=False, in_memory=False):
        """
        :param label_file: path to csv file containing labels for each and every image id of the data set.
        :param root_dir:  path to directory with all images.
        :param transform: optional transformation operation on each sample.
        :param greyscale: option to load data as monochrome images.
        :param in_memory: option to load data fully into memory.
        """
        self.data_frame = pd.read_csv(label_file)
        self.root_dir = root_dir
        self.transform = transform
        self.greyscale = greyscale
        self.in_memory = in_memory

        if in_memory:
            self.id2image = self._load_images()

    def _load_images(self):
        print('loading images in memory...')

        self.img_files = self.data_frame.id.to_list()

        id2image = {}
        Visualizer.printProgressBar(0, 1, prefix="Progress", suffix="Complete", length=50)
        for idx, file_name in enumerate(self.img_files):
            if idx % 220 == 0:
                Visualizer.printProgressBar(idx + 1, len(self.img_files), prefix="Progress", suffix="Complete", length=50)
            image = io.imread(fname=os.path.join(self.root_dir, file_name + ".tif"), as_gray=self.greyscale)
            id2image[file_name] = image

        print("Sucessfully loaded", len(id2image), "pictures into memory.")
        return id2image

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, index):
        """
         Fetch a data sample for a given key and turn it to greyscale.
        """
        if torch.is_tensor(index):
            index = index.tolist()

        if self.in_memory:
            file_name = self.img_files[index]
            image = self.id2image[file_name]
        else:
            img_path = os.path.join(self.root_dir, self.data_frame.iloc[index, 0])
            img_path = img_path + ".tif"
            image = io.imread(fname=img_path, as_gray=self.greyscale) # np ndarray

        label = self.data_frame.iloc[index, 1]
        label = label.astype(np.float32)

        if self.transform:
            image = self.transform(image)

        return image, label


class ToTensor(object):
    """Convert ndarray from sample to Tensor."""

    def __call__(self, image):
        # numpy image: H x W x Cß
        # torch image: C X H X W
        image = image.transpose((2, 0, 1)) # for colored images
        
        return torch.from_numpy(image)

class Normalize(object):
    """ Normalize a tensor(!) image with mean (type=sequence) and standard deviation (type=sequence) for each channel.
        Note: ToTensor() has to be applied first. """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        if type(image) != torch.Tensor:
            raise ValueError("Error: Normalize expects torch.Tensor image type.")
        image = image.float()
        image = transforms.Normalize(self.mean, self.std)(image)
        return image

class CenterCrop(object):
    """Crops the given Tensor Image at the center.

        Args:
            size (sequence or int): Desired output size of the crop. If size is an
                int instead of sequence like (h, w), a square crop (size, size) is
                made.

        Returns:
            Tensor Image: Cropped image.
        """
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        if type(image) != torch.Tensor:
            raise ValueError("Error: CenterCrop expects torch.Tensor image type.")

        # Convert Tensor image to PIL before
        image = transforms.ToPILImage()(image)
        # Perform CenterCrop
        image = transforms.CenterCrop(size=self.size)(image)
        # Convert PIL Image to Tensor after
        image = transforms.ToTensor()(image)
        return image

class RandomRotation(object):
    """ Rotate the image by angle """

    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, image):
        if type(image) != torch.Tensor:
            raise ValueError("Error: RandomRotation expects torch.Tensor image type.")

        # Convert Tensor image to PIL before
        image = transforms.ToPILImage()(image)
        # Perform RandomRotation
        image = transforms.RandomRotation((-180, 180))(image)
        # Convert PIL Image to Tensor after
        image = transforms.ToTensor()(image)
        return image

class  RandomHorizontalFlip(object):

    def __call__(self, image):
        if type(image) != torch.Tensor:
            raise ValueError("Error: RandomHorizontalFlip expects torch.Tensor image type.")

        # Convert Tensor image to PIL before
        image = transforms.ToPILImage()(image)

        # Perform RandomHorizontalFlip
        image = transforms.RandomHorizontalFlip()(image)

        # Convert PIL Image to Tensor after
        image = transforms.ToTensor()(image)
        return image
