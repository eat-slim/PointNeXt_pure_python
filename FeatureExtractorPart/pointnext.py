import torch
import torch.nn as nn
from FeatureExtractorPart.utils import index_points, farthest_point_sample, query_hybrid, \
    coordinate_distance, build_mlp


class SetAbstraction(nn.Module):
    """
    点云特征提取
    包含一个单尺度S-G-P过程
    """

    def __init__(self,
                 npoint: int,
                 radius: int,
                 nsample: int,
                 in_channel: int,
                 coor_dim: int = 3):
        """
        :param npoint: 采样点数量
        :param radius: 采样半径
        :param nsample: 采样点数量
        :param in_channel: 特征维度的输入值
        :param coor_dim: 点的坐标维度，默认为3
        """
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.in_channel = in_channel
        self.coor_dim = coor_dim
        self.mlp = build_mlp(in_channel=in_channel + coor_dim, channel_list=[in_channel * 2], dim=2)

    def forward(self, points_coor, points_fea):
        """
        :param points_coor: <torch.Tensor> (B, 3, N) 点云原始坐标
        :param points_fea: <torch.Tensor> (B, C, N) 点云特征
        :return:
            new_xyz: <torch.Tensor> (B, 3, S) 下采样后的点云坐标
            new_fea: <torch.Tensor> (B, D, S) 采样点特征
        """
        points_coor, points_fea = points_coor.permute(0, 2, 1), points_fea.permute(0, 2, 1)  # (B, C, N) -> (B, N, C)
        bs, nbr_point_in, _ = points_coor.shape
        num_point_out = self.npoint

        '''S 采样'''
        new_coor = index_points(points_coor, farthest_point_sample(points_coor, num_point_out))  # 获取新采样点 (B, S, coor)

        '''G 分组'''
        # 每个group的点云索引 (B, S, K)
        group_idx = query_hybrid(self.radius, self.nsample, points_coor[..., :3], new_coor[..., :3])

        # 基于分组获取各组内点云坐标和特征，并进行拼接
        grouped_points_coor = index_points(points_coor[..., :3], group_idx)  # 每个group内所有点云的坐标 (B, S, K, 3)
        grouped_points_coor -= new_coor[..., :3].view(bs, num_point_out, 1, 3)  # 坐标转化为与采样点的偏移量 (B, S, K, 3)
        grouped_points_coor = grouped_points_coor / self.radius  # 相对坐标归一化
        grouped_points_fea = index_points(points_fea, group_idx)  # 每个group内所有点云的特征 (B, S, K, C)
        grouped_points_fea = torch.cat([grouped_points_fea, grouped_points_coor], dim=-1)  # 拼接坐标偏移量 (B, S, K, C+3)

        '''P 特征提取'''
        # (B, S, K, C+3) -> (B, C+3, K, S) -mlp-> (B, D, K, S) -pooling-> (B, D, S)
        grouped_points_fea = grouped_points_fea.permute(0, 3, 2, 1)  # 2d卷积作用于维度1
        grouped_points_fea = self.mlp(grouped_points_fea)
        new_points_fea = torch.max(grouped_points_fea, dim=2)[0]

        new_coor = new_coor.permute(0, 2, 1)  # (B, 3, S)
        return new_coor, new_points_fea


class LocalAggregation(nn.Module):
    """
    局部特征提取
    包含一个单尺度G-P过程，每一个点都作为采样点进行group以聚合局部特征，无下采样过程
    """

    def __init__(self,
                 radius: int,
                 nsample: int,
                 in_channel: int,
                 coor_dim: int = 3):
        """
        :param radius: 采样半径
        :param nsample: 采样点数量
        :param in_channel: 特征维度的输入值
        :param coor_dim: 点的坐标维度，默认为3
        """
        super().__init__()
        self.radius = radius
        self.nsample = nsample
        self.in_channel = in_channel
        self.mlp = build_mlp(in_channel=in_channel + coor_dim, channel_list=[in_channel], dim=2)

    def forward(self, points_coor, points_fea):
        """
        :param points_coor: <torch.Tensor> (B, 3, N) 点云原始坐标
        :param points_fea: <torch.Tensor> (B, C, N) 点云特征
        :return:
            new_fea: <torch.Tensor> (B, D, N) 局部特征聚合后的特征
        """
        # (B, C, N) -> (B, N, C)
        points_coor, points_fea = points_coor.permute(0, 2, 1), points_fea.permute(0, 2, 1)
        bs, npoint, _ = points_coor.shape

        '''G 分组'''
        # 每个group的点云索引 (B, N, K)
        group_idx = query_hybrid(self.radius, self.nsample, points_coor[..., :3], points_coor[..., :3])

        # 基于分组获取各组内点云坐标和特征，并进行拼接
        grouped_points_coor = index_points(points_coor[..., :3], group_idx)  # 每个group内所有点云的坐标 (B, N, K, 3)
        grouped_points_coor = grouped_points_coor - points_coor[..., :3].view(bs, npoint, 1, 3)  # 坐标转化为与采样点的偏移量
        grouped_points_coor = grouped_points_coor / self.radius  # 相对坐标归一化
        grouped_points_fea = index_points(points_fea, group_idx)  # 每个group内所有点云的特征 (B, N, K, C)
        grouped_points_fea = torch.cat([grouped_points_fea, grouped_points_coor], dim=-1)  # 拼接坐标偏移量 (B, N, K, C+3)

        '''P 特征提取'''
        # (B, N, K, C+3) -> (B, C+3, K, N) -mlp-> (B, D, K, N) -pooling-> (B, D, N)
        grouped_points_fea = grouped_points_fea.permute(0, 3, 2, 1)  # 2d卷积作用于维度1
        grouped_points_fea = self.mlp(grouped_points_fea)
        new_fea = torch.max(grouped_points_fea, dim=2)[0]

        return new_fea


class InvResMLP(nn.Module):
    """
    逆瓶颈残差块
    """

    def __init__(self,
                 radius: int,
                 nsample: int,
                 in_channel: int,
                 coor_dim: int = 3,
                 expansion: int = 4):
        """
        :param radius: 采样半径
        :param nsample: 采样点数量
        :param in_channel: 特征维度的输入值
        :param coor_dim: 点的坐标维度，默认为3
        :param expansion: 中间层通道数扩张倍数
        """
        super().__init__()
        self.la = LocalAggregation(radius=radius, nsample=nsample, in_channel=in_channel, coor_dim=coor_dim)
        channel_list = [in_channel * expansion, in_channel]
        self.pw_conv = build_mlp(in_channel=in_channel, channel_list=channel_list, dim=1, drop_last_act=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, points):
        """
        :param points:
            <torch.Tensor> (B, 3, N) 点云原始坐标
            <torch.Tensor> (B, C, N) 点云特征
        :return:
            new_fea: <torch.Tensor> (B, D, N)
        """
        points_coor, points_fea = points
        identity = points_fea
        points_fea = self.la(points_coor, points_fea)
        points_fea = self.pw_conv(points_fea)
        points_fea = points_fea + identity
        points_fea = self.act(points_fea)
        return [points_coor, points_fea]


class Stage(nn.Module):
    """
    PointNeXt一个下采样阶段
    """

    def __init__(self,
                 npoint: int,
                 radius_list: list,
                 nsample_list: list,
                 in_channel: int,
                 coor_dim: int = 3,
                 expansion: int = 4):
        """
        :param npoint: 采样点数量
        :param radius_list: <list[float]> 采样半径
        :param nsample_list: <list[int]> 采样邻点数量
        :param in_channel: 特征维度的输入值
        :param coor_dim: 点的坐标维度，默认为3
        :param expansion: 中间层通道数扩张倍数
        """
        super().__init__()
        self.sa = SetAbstraction(npoint=npoint, radius=radius_list[0], nsample=nsample_list[0],
                                 in_channel=in_channel, coor_dim=coor_dim)

        irm = []
        for i in range(1, len(radius_list)):
            irm.append(
                InvResMLP(radius=radius_list[i], nsample=nsample_list[i], in_channel=in_channel * 2,
                          coor_dim=coor_dim, expansion=expansion)
            )
        self.irm = nn.Sequential(*irm)

    def forward(self, points_coor, points_fea):
        """
        :param points_coor: <torch.Tensor> (B, 3, N) 点云原始坐标
        :param points_fea: <torch.Tensor> (B, D, N) 点云特征
        :return:
            new_xyz: <torch.Tensor> (B, 3, S) 下采样后的点云坐标
            new_points_concat: <torch.Tensor> (B, D', S) 下采样后的点云特征
        """
        new_coor, new_points_fea = self.sa(points_coor, points_fea)
        new_coor, new_points_fea = self.irm([new_coor, new_points_fea])
        return new_coor, new_points_fea


class FeaturePropagation(nn.Module):
    """
    FP上采样模块
    """

    def __init__(self, in_channel, mlp, coor_dim=3):
        """
        :param in_channel: <list<int>> 同层和下层特征维度的输入值
        :param mlp: <list[int]> mlp的通道维度数
        """
        super(FeaturePropagation, self).__init__()
        self.mlp_modules = build_mlp(in_channel=sum(in_channel), channel_list=mlp, dim=1)
        self.coor_dim = coor_dim

    def forward(self, xyz1, xyz2, points1, points2):
        """
        :param xyz1: <torch.Tensor> (B, 3, N) 同层点云原始坐标
        :param xyz2: <torch.Tensor> (B, 3, S) 下层点云原始坐标
        :param points1: <torch.Tensor> (B, D1, N) 同层点云特征
        :param points2: <torch.Tensor> (B, D2, S) 下层点云特征
        :return: <torch.Tensor> (B, D, N) 上采样后的点云特征
        """
        B, _, N = xyz1.shape
        _, _, S = xyz2.shape

        if S == 1:
            # (B, D2, 1) -> (B, D2, N)  只有一个特征点则直接扩展至所有点
            new_points = points2.repeat(1, 1, N)
        else:
            # (B, C, N) -> (B, N, C)
            xyz1, xyz2, points2 = xyz1.permute(0, 2, 1), xyz2.permute(0, 2, 1), points2.permute(0, 2, 1)
            # 找到与每个同层点最近的前3个下层特征点
            dists = coordinate_distance(xyz1[..., :3], xyz2[..., :3])
            dists, idx = torch.topk(dists, k=3, dim=-1, largest=False)  # 基于距离选择最近点 (B, N, 3)

            # 基于距离进行特征值加权求和
            dist_recip = 1.0 / dists.clamp(min=1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)
            # (B, N, D2) -> (B, D2, N)
            new_points = interpolated_points.permute(0, 2, 1)

        # 下层特征值与同层特征值拼接，作为扩展后的特征值 (B, D2, N) -> (B, D1+D2, N) -> (B, D, N)
        new_points = torch.cat((points1, new_points), dim=1)
        new_points = self.mlp_modules(new_points)
        return new_points


class Head(nn.Module):
    """分类头 & 分割头"""
    def __init__(self, in_channel, mlp, num_class, task_type):
        """
        :param in_channel: <int> 特征维度的输入值
        :param mlp: <list[int]> mlp的通道维度数
        :param num_class: <int> 输出类别的数量
        """
        super(Head, self).__init__()
        mlp.append(num_class)
        self.mlp_modules = build_mlp(in_channel=in_channel, channel_list=mlp, dim=1,
                                     drop_last_norm_act=True, dropout=True)
        self.task_type = task_type

    def forward(self, points_fea):
        """
        :param points_fea: <torch.Tensor> (B, C, N) 点云特征
        :return: <torch.Tensor> (B, num_class, N) 点云特征
        """
        if self.task_type == 'classification':
            points_fea = torch.max(points_fea, dim=-1, keepdim=True)[0]  # (B, C, N) -> (B, C, 1)
        points_cls = self.mlp_modules(points_fea)
        return points_cls

