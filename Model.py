import torch.nn as nn
from FeatureExtractorPart.pointnext import Stage, FeaturePropagation, Head


class PointNeXt(nn.Module):
    """
    PointNeXt语义分割模型特征提取部分
    """

    def __init__(self, cfg):
        super().__init__()
        self.type = cfg['type']
        self.num_class = cfg['num_class']
        self.coor_dim = cfg['coor_dim']
        self.normal = cfg['normal']
        width = cfg['width']

        self.mlp = nn.Conv1d(in_channels=self.coor_dim + self.coor_dim * self.normal,
                             out_channels=width, kernel_size=1)
        self.stage = nn.ModuleList()

        for i in range(len(cfg['npoint'])):
            self.stage.append(
                Stage(
                    npoint=cfg['npoint'][i], radius_list=cfg['radius_list'][i], nsample_list=cfg['nsample_list'][i],
                    in_channel=width, expansion=cfg['expansion'], coor_dim=self.coor_dim
                )
            )
            width *= 2

        if self.type == 'segmentation':
            self.decoder = nn.ModuleList()
            for i in range(len(cfg['npoint'])):
                self.decoder.append(
                    FeaturePropagation(in_channel=[width, width // 2], mlp=[width // 2, width // 2],
                                       coor_dim=self.coor_dim)
                )
                width = width // 2

        self.head = Head(in_channel=width, mlp=cfg['head'], num_class=self.num_class, task_type=self.type)

    def forward(self, x):
        l0_xyz, l0_points = x[:, :self.coor_dim, :], x[:, :self.coor_dim + self.coor_dim * self.normal, :]
        l0_points = self.mlp(l0_points)

        record = [[l0_xyz, l0_points]]
        for stage in self.stage:
            record.append(list(stage(*record[-1])))
        if self.type == 'segmentation':
            for i, decoder in enumerate(self.decoder):
                record[-i-2][1] = decoder(record[-i-2][0], record[-i-1][0], record[-i-2][1], record[-i-1][1])
            points_cls = self.head(record[0][1])

        else:
            points_cls = self.head(record[-1][1])

        return points_cls

