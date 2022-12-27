from Parameters import *
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_index
sys.path.insert(1, os.path.dirname(os.path.abspath(__name__)))
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
import torch
from Model import PointNeXt
from dataset.ModelNet40 import ModelNet40
from Loss import LabelSmoothingCE
from Transforms import PCDPretreatment, get_data_augment
from Trainer import Trainer
from utils import IdentityScheduler


def main():
    # 解析参数
    if args.use_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        gpus = list(range(torch.cuda.device_count()))
        torch.cuda.set_device('cuda:{}'.format(gpus[0]))
    else:
        args.device = torch.device('cpu')

    if sys.platform == 'darwin':
        args.num_workers = 0
        args.batch_size = 2

    model_cfg = MODEL_CONFIG[args.model_cfg]
    max_input = model_cfg['max_input']
    normal = model_cfg['normal']

    if args.optimizer.lower() == 'adamw':
        Optimizer = torch.optim.AdamW
    else:
        args.optimizer = 'Adam'
        Optimizer = torch.optim.Adam

    if args.scheduler.lower() == 'identity':
        Scheduler = IdentityScheduler
    else:
        args.scheduler = 'cosine'
        Scheduler = torch.optim.lr_scheduler.CosineAnnealingLR

    # 数据变换、加载数据集
    logger.info('Prepare Data')
    '''数据变换、加载数据集'''
    data_augment, random_sample, random_drop = get_data_augment(DATA_AUG_CONFIG[args.data_aug])
    transforms = PCDPretreatment(num=max_input, down_sample='fps', normal=normal,
                                 data_augmentation=data_augment, random_drop=random_drop, resampling=random_sample)

    '''Prepare dataset'''
    if args.dataset_path is None or args.dataset_path == 'default':
        if args.dataset == 'ModelNet40':
            default_dataset_path_list = [r'../../Dataset/ModelNet40_points',
                                         r'/Users/dingziheng/dataset/ModelNet40_points',
                                         r'/root/dataset/ModelNet40_points']
        else:
            raise ValueError
        for path in default_dataset_path_list:
            if os.path.exists(path):
                args.dataset_path = path
                break
        else:  # this is for-else block, indent is not missing
            raise FileNotFoundError(f'Dataset path not found.')
        logger.info(f'Load default dataset from {args.dataset_path}')
    if args.dataset == 'ModelNet40':
        dataset = ModelNet40(dataroot=args.dataset_path, transforms=transforms)
    else:
        raise ValueError

    # 模型与损失函数
    logger.info('Prepare Models...')
    model = PointNeXt(model_cfg).to(device=args.device)
    optimizer = Optimizer(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = Scheduler(optimizer, T_max=args.num_epochs, eta_min=args.lr * 0.001)
    criterion = LabelSmoothingCE()

    # 训练器
    logger.info('Trainer launching...')
    trainer = Trainer(
        args=args,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        dataset=dataset,
        mode=args.mode
    )
    trainer.run()


if __name__ == "__main__":
    main()
    print('Done.')
