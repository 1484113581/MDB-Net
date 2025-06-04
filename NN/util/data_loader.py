# -*- coding: utf-8 -*-
# @Time    : 2024/4/23 19:44
# @Author  : CHEN
import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
# import albumentations as album
import tifffile as tif
# from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image

import torchvision.transforms as transforms


def find_files_with_extension(folder_path, extension):
    """
    查找指定文件夹下以特定后缀结尾的所有文件。并进行转换，例如将 "s1_baoding_16.tif" 转成 "_baoding_16.tif"

    Args:
        folder_path (str): 文件夹路径。
        extension (str): 目标后缀（例如 '.txt'）。

    Returns:
        list: 包含匹配文件名的列表。
    """
    matching_files = []
    for filename in os.listdir(folder_path):
        if filename.endswith(extension):
            matching_files.append(filename.split("_", 1)[1])
    return matching_files


# 有strm、boundary
class BHDataset14(Dataset):
    def __init__(self, datasetPath, augmentations=False):
        """
        :param datasetPath: 数据集路径
        :param augmentations:是否数据增强
        """
        self.datasetPath = datasetPath
        self.augmentations = augmentations

        # 获取每个样本的路径，并存储到列表中
        self.file_path_suffix = find_files_with_extension(os.path.join(datasetPath, 's1'), '.tif')

    def __getitem__(self, index):
        s1_path = os.path.join(self.datasetPath, 's1', 's1_' + self.file_path_suffix[index])
        s2_path = os.path.join(self.datasetPath, 's2', 's2_' + self.file_path_suffix[index])
        palsar_path = os.path.join(self.datasetPath, 'palsar', 'palsar_' + self.file_path_suffix[index])
        strm_path = os.path.join(self.datasetPath, 'strm', 'strm_' + self.file_path_suffix[index])
        bh_path = os.path.join(self.datasetPath, 'bh', 'bh_' + self.file_path_suffix[index])
        fp_path = os.path.join(self.datasetPath, 'fp', 'fp_' + self.file_path_suffix[index])
        boundary_path = os.path.join(self.datasetPath, 'boun', 'boun_' + self.file_path_suffix[index])

        # -------------------特征集------------
        # Read feature TIFF files
        s1 = tif.imread(s1_path)
        s2 = tif.imread(s2_path)
        palsar = tif.imread(palsar_path)
        strm = tif.imread(strm_path)

        s1 = np.transpose(s1, (2, 0, 1))
        s2 = np.transpose(s2, (2, 0, 1))
        palsar = np.transpose(palsar, (2, 0, 1))
        strm = strm.reshape(1, 128, 128)

        s1 = torch.tensor(s1, dtype=torch.float32)
        s2 = torch.tensor(s2, dtype=torch.float32)
        palsar = torch.tensor(palsar.astype(np.float32), dtype=torch.float32)
        strm = torch.tensor(strm, dtype=torch.float32)

        # 直接先归一化吧
        data_mean_s2 = [8.8033e-02, 1.0361e-01, 1.0447e-01, 1.7948e-01]
        data_std_s2 = [3.0823e-02, 3.2570e-02, 3.9694e-02, 5.5232e-02]
        data_normalize_s2 = transforms.Normalize(mean=data_mean_s2, std=data_std_s2)
        s2 = data_normalize_s2(s2)

        data_mean_s1 = [-8.0055e+00, -1.5418e+01]
        data_std_s1 = [3.8221e+00, 3.4108e+00]
        data_normalize_s1 = transforms.Normalize(mean=data_mean_s1, std=data_std_s1)
        s1 = data_normalize_s1(s1)

        data_mean_palsar = [7.8584e+03, 2.8027e+03]
        data_std_palsar = [4.6073e+03, 1.4685e+03]
        data_normalize_palsar = transforms.Normalize(mean=data_mean_palsar, std=data_std_palsar)
        palsar = data_normalize_palsar(palsar)

        data_mean_strm = [1.1333e+02]
        data_std_strm = [1.0113e+01]
        data_normalize_strm = transforms.Normalize(mean=data_mean_strm, std=data_std_strm)
        strm = data_normalize_strm(strm)

        # Merge bands
        # feature_bands = np.concatenate((s2,mbi, s1,vvh ,palsar,strm), axis=0)
        # feature_bands = np.transpose(feature_bands, (2, 0, 1))  # Convert to channels-first format
        feature_bands = torch.cat((s2, s1, palsar, strm), dim=0)

        # Apply data augmentation if specified
        if self.augmentations:
            feature_bands = data_transfroms(image=feature_bands)['image']

        # Convert to tensor
        # feature_bands = torch.tensor(feature_bands, dtype=torch.float32)

        # -------------------标签------------
        buildingHeight = tif.imread(bh_path)
        h, w = buildingHeight.shape
        buildingHeight = buildingHeight.reshape(1, h, w)
        buildingHeight = torch.tensor(buildingHeight, dtype=torch.float32)

        footprint = tif.imread(fp_path)
        footprint = footprint.reshape(1, h, w)
        footprint = torch.tensor(footprint, dtype=torch.float32)

        boundary = tif.imread(boundary_path)
        boundary = boundary.reshape(1, 128, 128)
        boundary = torch.tensor(boundary, dtype=torch.float32)

        sample = {"feature": feature_bands, "footprint": footprint, "height": buildingHeight, "boundary": boundary}

        return sample

    def __len__(self):
        return len(self.file_path_suffix)


def load_data_new(filepath, batchsize):
    # 加载数据集
    train_dataset = BHDataset14(os.path.join(filepath, "train"))
    train_iter = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
    num_trian = len(train_dataset)

    val_dataset = BHDataset14(os.path.join(filepath, "val"))
    val_iter = DataLoader(val_dataset, batch_size=batchsize, shuffle=False)
    num_val = len(val_dataset)
    return train_iter, val_iter, num_trian, num_val


if __name__ == '__main__':
    pass
