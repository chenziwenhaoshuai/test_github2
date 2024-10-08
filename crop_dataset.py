from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import torch
from torchvision.transforms import Resize, ToTensor

LABEL_SIZE = (480, 640)  # (640,960)


class DataSet_FCN(Dataset):
    def __init__(self, root, file, data_transform, label_transform, mode, label_size):
        self.file = file
        self.root = root
        self.mode = mode
        # self.image_files = np.array([x.path for x in os.scandir(root)
        #                              if x.name.endswith(".jpg") or x.name.endswith(".png") or x.name.endswith(".JPG")])
        self.data_transform = data_transform
        self.label_transform = label_transform
        self.label_size = label_size

        self.img_list, self.label_png_list = self.analyses_txt()

    def __getitem__(self, item):

        img = Image.open(self.img_list[item])
        img = self.data_transform(img)
        if self.mode == 'train':
            # label = label_png_list[item]
            label = self.creat_label_png(None, self.label_png_list[item])
            label = np.resize(label, self.label_size)
            label = self.label_transform(label)
            label = torch.squeeze(label).long()
        else:
            label = 0
            if "yes" in self.img_list[item]:
                label = 1

        return img, label

    def __len__(self):
        # self.img_list, label_png_list = self.analyses_txt()
        return len(self.img_list)

    # 将标注转尺寸为图片大小的矩阵
    def creat_label_png(self, img_path, label_list):
        # image = Image.open(self.root + img_path)
        # width, height = image.size
        width, height = 640, 480
        label_png = np.zeros((height, width))

        label_png[0:int(height / 2), 0:int(width / 2)] = label_list[0]
        label_png[0:int(height / 2), int(width / 2):width] = label_list[1]
        label_png[int(height / 2):height, 0:int(width / 2)] = label_list[2]
        label_png[int(height / 2):height, int(width / 2):width] = label_list[3]

        return label_png

    ## 解析数据列表，转换格式之后的label数据列表，并获取图片数据列表
    def analyses_txt(self):
        with open(self.file, 'r') as f:
            img_label_list = f.readlines()
            f.close()
        img_list = []
        label_png_list = []
        for list in img_label_list:
            img_path, label_list = list.strip().split()
            label_list = label_list.split(',')
            # label_png_list.append(self.creat_label_png(img_path, label_list))
            label_png_list.append(label_list)
            img_list.append(self.root + img_path)
        return img_list, np.array(label_png_list)