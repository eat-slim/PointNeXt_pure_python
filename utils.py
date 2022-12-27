import torch
import numpy as np
import open3d as o3d


class fakecast:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class MetricLogger(object):
    def __init__(self) -> None:
        self.metrics = dict()

    def add_metric(self, name: str, value):
        assert (isinstance(value, int) or isinstance(value, float)
                or (isinstance(value, np.ndarray) and value.ndim in [1,2])
                or (isinstance(value, torch.Tensor) and value.ndim == [1,2]))
        if (name not in self.metrics.keys()):
            self.metrics[name] = [value]
        else:
            self.metrics[name].append(value)

    def clear(self):
        self.metrics = dict()

    def tostring(self):
        result = self.get_average_value()
        s = ''
        for name, mean_value in result.items():
            s += f'| {name} = '
            if (isinstance(mean_value, int) or isinstance(mean_value, float)):
                s += f'{mean_value:>5.3f} '
            elif (isinstance(mean_value, np.ndarray)):
                for v in mean_value:
                    s += f'{float(v):>5.3f}, '
            elif (isinstance(mean_value, torch.Tensor)):
                for v in mean_value:
                    s += f'{float(v):>5.3f}, '
        return s

    def get_average_value(self):
        result = dict()
        for name, values in self.metrics.items():
            if (isinstance(values[0], int) or isinstance(values[0], float)):
                meanval = sum(values) / len(values)
            elif (isinstance(values[0], np.ndarray)):
                meanval = np.stack(values, axis=0).mean(0)
            elif (isinstance(values[0], torch.tensor)):
                meanval = torch.stack(values, dim=0).mean(0)
            result[name] = meanval
        return result


def GetPcdFromNumpy(pcd_np: np.ndarray, color=None):
    '''
    convert a numpy.ndarray with shape(xyz+, n) to pointcloud in o3d
    '''
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_np[:3, :].T)
    if (color is not None):
        pcd.paint_uniform_color(color)
    return pcd


def show_pcd(pcds, colors=None, normal=False, window_name="PCD"):
    '''
    pcds: List(ArrayLike), points to be shown, shape (K, xyz+)
    colors: List[Tuple], color list, shape (r,g,b) scaled 0~1
    '''
    import open3d as o3d
    # 创建窗口对象
    vis = o3d.visualization.Visualizer()
    # 设置窗口标题
    vis.create_window(window_name=window_name)
    # 设置点云大小
    # vis.get_render_option().point_size = 1
    # 设置颜色背景为黑色
    # opt = vis.get_render_option()
    # opt.background_color = np.asarray([0, 0, 0])

    for i in range(len(pcds)):
        # 创建点云对象
        pcd_o3d = o3d.open3d.geometry.PointCloud()
        # 将点云数据转换为Open3d可以直接使用的数据类型
        if (isinstance(pcds[i], np.ndarray)):
            pcd_points = pcds[i][:, :3]
            if normal:
                pcd_normals = pcds[i][:, 3:6]
        elif (isinstance(pcds[i], torch.Tensor)):
            pcd_points = pcds[i][:, :3].detach().cpu().numpy()
            if normal:
                pcd_normals = pcds[i][:, 3:6].detach().cpu().numpy()
        else:
            pcd_points = np.array(pcds[i][:, :3])
            if normal:
                pcd_normals = np.array(pcds[i][:, 3:6])
        pcd_o3d.points = o3d.open3d.utility.Vector3dVector(pcd_points)
        if normal:
            pcd_o3d.normals = o3d.open3d.utility.Vector3dVector(pcd_normals)
        # 设置点的颜色
        if colors is not None:
            pcd_o3d.paint_uniform_color(colors[i])
        # 将点云加入到窗口中
        vis.add_geometry(pcd_o3d)

    vis.run()
    vis.destroy_window()


def voxel_down_sample(pcd, voxel_size=0.3, num=None, padding=True):
    """
    点云体素化下采样
    :param pcd: <torch.Tensor> (N, 3+) 原始点云
    :param voxel_size: <float> 体素边长
    :param num: <int> 下采样后的点数数量，为None时不约束，否则采样后点云数量不超过该值
    :param padding: <int> num不为None且采样后数量不足时，使用0填充
    :return: <torch.Tensor> (N, 3+) 下采样后的点云
    """
    pcd_xyz = pcd[:, :3]
    # 根据点云范围确定voxel数量
    xyz_min = torch.min(pcd_xyz, dim=0)[0]
    xyz_max = torch.max(pcd_xyz, dim=0)[0]
    X, Y, Z = torch.div(xyz_max[0] - xyz_min[0], voxel_size, rounding_mode='trunc') + 1, \
              torch.div(xyz_max[1] - xyz_min[1], voxel_size, rounding_mode='trunc') + 1, \
              torch.div(xyz_max[2] - xyz_min[2], voxel_size, rounding_mode='trunc') + 1

    # 计算每个点云所在voxel的xyz索引和总索引
    relative_xyz = pcd_xyz - xyz_min
    voxel_xyz = torch.div(relative_xyz, voxel_size, rounding_mode='trunc').int()
    voxel_id = (voxel_xyz[:, 0] + voxel_xyz[:, 1] * X + voxel_xyz[:, 2] * X * Y).int()

    '''每个voxel仅保留最接近中心点的点云，并根据voxel内点云数量排序'''
    dis = torch.sum((relative_xyz - voxel_xyz * voxel_size - voxel_size / 2).pow(2), dim=-1)

    # 预先根据点云距离voxel中心的距离由近到远进行排序，使得每个voxel第一次被统计时即对应了最近点云
    dis, sorted_id = torch.sort(dis)
    voxel_id = voxel_id[sorted_id]
    pcd = pcd[sorted_id]

    # 去除相同voxel，id即为每个voxel内的采样点，cnt为当前采样点所在voxel的点云数量之和
    _, unique_id, cnt = np.unique(voxel_id.cpu(), return_index=True, return_counts=True)
    unique_id, cnt = torch.tensor(unique_id, device=pcd.device), torch.tensor(cnt, device=pcd.device)

    # 保留点云数量最多的voxel
    if num is not None and unique_id.shape[0] > num:
        _, cnt_topk_id = torch.topk(cnt, k=num)
        unique_id = unique_id[cnt_topk_id]
    new_pcd = pcd[unique_id]

    if num is not None:
        if new_pcd.shape[0] < num and padding:
            padding_num = num - new_pcd.shape[0]
            padding = torch.zeros(size=(padding_num, new_pcd.shape[1]), device=new_pcd.device)
            padding[:, 2] = -10
            new_pcd = torch.cat((new_pcd, padding), dim=0)
        else:
            new_pcd = new_pcd[:num]

    return new_pcd


class IdentityScheduler(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def step(self):
        pass

