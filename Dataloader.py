import cv2
from PIL import Image
import numpy as np
import os
from torchvision import transforms, datasets
from torch.utils import data
import torch


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class BUS_Dataset(data.Dataset):
    def __init__(self, base_dir="dataset", mode="train"):
        "Initiliaztion"
        self.base_dir = base_dir
        self.ten_trans = transforms.Compose([transforms.ToTensor()])
        self.mode = mode
        self.file_names = []

        if mode == "val":
            for file in sorted(os.listdir(os.path.join(base_dir, "images/"))):
                if file[-6] == "1":
                    self.file_names.append(os.path.join(base_dir, "images/", file))
        elif mode == "test":
            for file in sorted(os.listdir(os.path.join(base_dir, "images/"))):
                if file[-6] == "2":
                    self.file_names.append(os.path.join(base_dir, "images/", file))
        else:
            for file in sorted(os.listdir(os.path.join(base_dir, "images/"))):
                if (file[-6] != "1") and (file[-6] != "2"):
                    self.file_names.append(os.path.join(base_dir, "images/", file))

    def __len__(self):
        "Return total number of samples in data set"
        return len(self.file_names)

    def __getitem__(self, index):

        file_name = self.file_names[index]

        # stage 0 is input,x
        x = Image.open(file_name.replace("images", "stage_0"))
        x = self.ten_trans(x)

        # images are the output, y
        y = Image.open(file_name)

        y = self.ten_trans(y)

        return x, y, file_name
