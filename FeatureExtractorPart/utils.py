import torch
import torch.nn as nn


def build_mlp(in_channel, channel_list, dim=2, bias=False, drop_last_act=False,
              drop_last_norm_act=False, dropout=False):
    """
    构造基于n dim 1x1卷积的mlp
    :param in_channel: <int> 特征维度的输入值
    :param channel_list: <list[int]> mlp各层的输出通道维度数
    :param dim: <int> 维度，1或2
    :param bias: <bool> 卷积层是否添加bias，一般BN前的卷积层不使用bias
    :param drop_last_act: <bool> 是否去除最后一层激活函数
    :param drop_last_norm_act: <bool> 是否去除最后一层标准化层和激活函数
    :param dropout: <bool> 是否添加dropout层
    :return: <torch.nn.ModuleList[torch.nn.Sequential]>
    """
    # 解析参数获取相应卷积层、归一化层、激活函数
    if dim == 1:
        Conv = nn.Conv1d
        NORM = nn.BatchNorm1d
    else:
        Conv = nn.Conv2d
        NORM = nn.BatchNorm2d
    ACT = nn.ReLU

    # 根据通道数构建mlp
    mlp = []
    for i, channel in enumerate(channel_list):
        if dropout and i > 0:
            mlp.append(nn.Dropout(0.5, inplace=False))
        # 每层为conv-bn-relu
        mlp.append(Conv(in_channels=in_channel, out_channels=channel, kernel_size=1, bias=bias))
        mlp.append(NORM(channel))
        mlp.append(ACT(inplace=True))
        if i < len(channel_list) - 1:
            in_channel = channel

    if drop_last_act:
        mlp = mlp[:-1]
    elif drop_last_norm_act:
        mlp = mlp[:-2]
        mlp[-1] = Conv(in_channels=in_channel, out_channels=channel, kernel_size=1, bias=True)

    return nn.Sequential(*mlp)


def coordinate_distance(src, dst):
    """
    计算两个点集的各点间距
    !!!使用半精度运算或自动混合精度时[不要]使用化简的方法，否则会出现严重的浮点误差
    :param src: <torch.Tensor> (B, M, C) C为坐标
    :param dst: <torch.Tensor> (B, N, C) C为坐标
    :return: <torch.Tensor> (B, M, N)
    """
    B, M, _ = src.shape
    _, N, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.transpose(1, 2))
    dist += torch.sum(src ** 2, -1).view(B, M, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, N)

    # dist = torch.sum((src.unsqueeze(2) - dst.unsqueeze(1)).pow(2), dim=-1)
    return dist


def index_points(points, idx):
    """
    跟据采样点索引获取其原始点云xyz坐标等信息
    :param points: <torch.Tensor> (B, N, 3+) 原始点云
    :param idx: <torch.Tensor> (B, S)/(B, S, G) 采样点索引，S为采样点数量，G为每个采样点grouping的点数
    :return: <torch.Tensor> (B, S, 3+)/(B, S, G, 3+) 获取了原始点云信息的采样点
    """
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=points.device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    最远点采样
    随机选择一个初始点作为采样点，循环的将与当前采样点距离最远的点当作下一个采样点，直至满足采样点的数量需求
    :param xyz: <torch.Tensor> (B, N, 3+) 原始点云
    :param npoint: <int> 采样点数量
    :return: <torch.Tensor> (B, npoint) 采样点索引
    """
    device = xyz.device
    B, N, C = xyz.shape
    npoint = min(npoint, N)
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10  # 每个点与最近采样点的最小距离
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)  # 随机选取初始点

    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, -1)  # [bs, 1, coor_dim]
        dist = torch.nn.functional.pairwise_distance(xyz, centroid)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_hybrid(radius, nsample, xyz, new_xyz):
    """
    基于采样点进行KNN与ball query混合的grouping
    :param radius: <float> grouping半径
    :param nsample: <int> group内点云数量
    :param xyz: <torch.Tensor> (B, N, 3) 原始点云
    :param new_xyz: <torch.Tensor> (B, S, 3) 采样点
    :return: <torch.Tensor> (B, S, nsample) 每个采样点grouping的点云索引
    """
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape

    dist = coordinate_distance(new_xyz, xyz)  # 每个采样点与其他点的距离的平方
    dist, group_idx = torch.topk(dist, k=nsample, dim=-1, largest=False)  # 基于距离选择最近的作为采样点
    radius = radius ** 2
    mask = dist > radius  # 距离较远的点替换为距离最近的点
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    group_idx[mask] = group_first[mask]

    return group_idx





