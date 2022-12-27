import torch
import torch.nn as nn
from torchvision.transforms import Compose
import math
import random
from utils import voxel_down_sample
from FeatureExtractorPart.utils import index_points, farthest_point_sample


class PCDPretreatment(nn.Module):
    """
    点云预处理与部分数据增强
    """

    def __init__(self, num=2048, padding=True, down_sample='fps', mode='train', normal=True,
                 data_augmentation=None, random_drop=0, resampling=False):
        super().__init__()
        self.num = num
        self.padding = padding
        self.normal = normal
        self.random_drop = random_drop
        self.resampling = resampling
        self.mode = mode
        self.sampling = down_sample
        self.set_sampling(down_sample)
        self.data_aug = data_augmentation if data_augmentation is not None else nn.Identity()

    def forward(self, pcd):
        """
        :param pcd: <torch.Tensor> (N, 3+) 点云矩阵
        :return: <torch.Tensor> (3+, N)
        """
        # 坐标归一化
        pcd_xyz = pcd[:, :3]
        pcd_xyz = pcd_xyz - pcd_xyz.mean(dim=0, keepdim=True)
        dis = torch.norm(pcd_xyz, dim=1)
        max_dis = dis.max()
        pcd_xyz /= max_dis

        # 法线
        if self.normal:
            pcd[:, :3] = pcd_xyz
        else:
            pcd = pcd_xyz

        # 随机丢弃一定比率的点
        if self.random_drop > 0 and self.mode == 'train':
            drop_ratio = random.uniform(0, self.random_drop)
            remain_points = torch.rand(size=(pcd.shape[0],), device=pcd.device) >= drop_ratio
            pcd = pcd[remain_points]

        # 点云数量统一化
        if pcd.shape[0] < self.num and self.padding:
            padding = torch.zeros(size=(self.num - pcd.shape[0], pcd.shape[1]), device=pcd.device)
            padding[:, 2] = -10
            pcd = torch.cat((pcd, padding), dim=0)
        elif pcd.shape[0] > self.num:
            pcd = self.down_sample(pcd)
        pcd = pcd.T

        # 点云数量无关的数据增强
        if self.mode == 'train':
            if self.normal:
                pcd[:3, :] = self.data_aug(pcd[:3, :])
            else:
                pcd[:6, :] = self.data_aug(pcd[:6, :])

        return pcd

    def min_dis(self, pcd):
        """
        基于距离的下采样，保留距离车辆最近的点
        """
        dis = torch.norm(pcd[:, :3], p=2, dim=1)  # 计算点云距离车辆的直线距离
        _, sorted_ids = torch.sort(dis)
        sorted_ids = sorted_ids[:self.num]  # 只保留距离最近的num个点
        pcd = pcd[sorted_ids]
        return pcd

    def random(self, pcd):
        """
        随机点云下采样
        """
        downsample_ids = torch.randperm(pcd.shape[0])[:self.num]
        pcd = pcd[downsample_ids]
        return pcd

    def voxel_down_sample(self, pcd):
        return voxel_down_sample(pcd, voxel_size=0.01, num=self.num, padding=self.padding)

    def fps(self, pcd):
        sample_ids = farthest_point_sample(pcd[:, :3].unsqueeze(0), self.num)
        pcd = index_points(pcd.unsqueeze(0), sample_ids)[0]
        return pcd

    def set_padding(self, option: bool):
        self.padding = option

    def set_sampling(self, sampling):
        if sampling == 'dis':
            self.down_sample = self.min_dis
        elif sampling == 'voxel':
            self.down_sample = self.voxel_down_sample
        elif sampling == 'random':
            self.down_sample = self.random
        elif sampling == 'fps':
            self.down_sample = self.fps
        elif sampling == 'identical':
            self.down_sample = lambda x: x
        else:
            raise ValueError

    def set_mode(self, mode):
        self.mode = mode
        if self.resampling:
            if mode == 'train':
                self.down_sample = self.random
            else:
                self.set_sampling(self.sampling)


class RandomRT(nn.Module):
    def __init__(self, r_mean=0, r_std=0.5, t_mean=0, t_std=0.1, p=1) -> None:
        super().__init__()
        self.r_mean = r_mean
        self.r_std = r_std
        self.t_mean = t_mean
        self.t_std = t_std
        self.p = p

    def forward(self, input):
        """
            pcd: Tensor [3+, n]
        """
        if random.random() > self.p:
            return input
        pcd = input

        # 生成三方向的随机角度，得到各方位的旋转矩阵，最后整合为总体旋转矩阵
        z = (torch.rand(size=(1,)) - 0.5) * 2 * self.r_std
        x = (torch.rand(size=(1,)) - 0.5) * 2 * self.r_std
        y = (torch.rand(size=(1,)) - 0.5) * 2 * self.r_std

        R_x = torch.tensor([[1, 0, 0],
                            [0, math.cos(x), -math.sin(x)],
                            [0, math.sin(x), math.cos(x)]])
        R_y = torch.tensor([[math.cos(y), 0, math.sin(y)],
                            [0, 1, 0],
                            [-math.sin(y), 0, math.cos(y)]])
        R_z = torch.tensor([[math.cos(z), -math.sin(z), 0],
                            [math.sin(z), math.cos(z), 0],
                            [0, 0, 1]])

        R_aug = R_x @ R_y @ R_z
        R_aug.to(pcd.device)

        if self.t_std > 0:
            T_aug = (torch.rand(size=(3, 1)) - 0.5) * 2 * self.t_std
        else:
            T_aug = torch.zeros(size=(3, 1), device=pcd.device)

        pcd[:3, :] = R_aug @ pcd[:3, :] + T_aug
        if pcd.shape[0] >= 6:
            pcd[3:6, :] = R_aug @ pcd[3:6, :]

        return pcd


class RandomPosJitter(nn.Module):
    """点云位置随机抖动"""
    def __init__(self, mean=0, std=0.01, p=1):
        super().__init__()
        self.mean = mean
        self.std = std
        self.p = p

    def forward(self, input):
        """
        :param input:
            pcd: Tensor [3+, n]
        :return:
        """
        if random.random() > self.p:
            return input
        pcd = input
        pos_jitter = (torch.rand(size=(3, pcd.shape[1])) - 0.5) * 2 * self.std
        pcd[:3, :] += pos_jitter
        return pcd


def get_data_augment(data_aug):
    aug_list, data_augment, random_sample, random_drop = [], None, False, 0
    if 'RT' in data_aug and data_aug['RT'] is not None:
        aug_list.append(RandomRT(*data_aug['RT']))
    if 'jitter' in data_aug and data_aug['jitter'] is not None:
        aug_list.append(RandomPosJitter(*data_aug['jitter']))
    if 'random_sample' in data_aug and data_aug['random_sample'] is not None:
        random_sample = data_aug['random_sample']
    if 'random_drop' in data_aug and data_aug['random_drop'] is not None:
        random_drop = data_aug['random_drop']

    if len(aug_list) == 1:
        data_augment = aug_list[0]
    elif len(aug_list) > 1:
        data_augment = Compose(aug_list)
    return data_augment, random_sample, random_drop
