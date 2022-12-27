import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast as autocast
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
from utils import MetricLogger, fakecast
from utils import show_pcd

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.manual_seed(3407)


class Trainer:
    """
    训练器，输入待训练的模型、参数，封装训练过程
    """
    def __init__(self, args, model, optimizer, scheduler, criterion, dataset, mode):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.dataset = dataset
        self.mode = mode
        self.dataloader = None
        self.epoch = 1
        self.step = 1

        self.epoch_metric_logger = MetricLogger()

        # 恢复检查点
        if self.args.checkpoint != '':
            checkpoint = torch.load(self.args.checkpoint, map_location=self.args.device)
            # Load model
            if 'model' in checkpoint:
                model_state_dict = checkpoint['model']
                if model_state_dict.keys() != self.model.state_dict().keys():
                    logger.info("Load model Failed, keys not match..")
                else:
                    self.model.load_state_dict(model_state_dict)
                    logger.info("Load model state")
                    if 'optimizer' in checkpoint:
                        self.optimizer.load_state_dict(checkpoint['optimizer'])
                        logger.info("Load optimizer state")
                    if 'scheduler' in checkpoint:
                        self.scheduler.load_state_dict(checkpoint['scheduler'])
                        logger.info("Load scheduler state")

            if 'epoch' in checkpoint:
                self.epoch = checkpoint['epoch'] + 1
                logger.info(f"Load epoch, current = {self.epoch}")

            if 'step' in checkpoint:
                self.step = checkpoint['step'] + 1
                logger.info(f"Load step, current = {self.step}")
            logger.info(f'Load checkpoint complete: \'{self.args.checkpoint}\'')
        else:
            logger.info(f'{mode} with a initial model')

        if self.args.auto_cast:
            self.cast = autocast
        else:
            self.cast = fakecast

        # 创建训练、测试结果保存目录
        self.log = f'{self.args.name}_model={self.args.model_cfg}_ds={self.args.dataset}_aug={self.args.data_aug}_' \
                   f'lr={self.args.lr}_wd={self.args.wd}_bs={self.args.batch_size}_' \
                   f'{self.args.optimizer}_{self.args.scheduler}'
        if self.mode == 'train':
            self.save_root = os.path.join('./result_train', self.log)
        elif self.mode == 'test':
            self.save_root = os.path.join('./result_test', self.log)
        else:
            raise ValueError
        os.makedirs(self.save_root, exist_ok=True)
        logger.info(f'save root = \'{self.save_root}\'')
        logger.info(f'run in {self.args.device}')

    def run(self):
        if self.mode == 'train':
            self.train()
        elif self.mode == 'test':
            self.test()

    def train(self):
        # tensorboard可视化训练过程，记录训练时的相关数据，使用指令:tensorboard --logdir=runs
        self.writer = SummaryWriter(os.path.join('./runs', self.log))

        self.dataloader = DataLoader(dataset=self.dataset,
                                     batch_size=self.args.batch_size,
                                     num_workers=self.args.num_workers,
                                     shuffle=True,
                                     pin_memory=True,
                                     drop_last=False)

        start_epoch = self.epoch
        for ep in range(start_epoch, self.args.num_epochs + 1):
            # 记录日志
            self.writer.add_scalar("learning_rate", self.optimizer.param_groups[0]['lr'], ep)

            # 单轮训练
            self.train_one_epoch()

            # 动态学习率
            self.scheduler.step()

            # 定期保存
            if self.epoch % self.args.save_cycle == 0:
                self.save()

            # 定期验证
            if self.epoch % self.args.eval_cycle == 0:
                self.test()

            self.epoch += 1

        self.save(finish=True)

    def train_one_epoch(self):
        self.model.train()
        self.dataset.train()
        epoch_loss, epoch_acc = [], []
        count = self.args.log_cycle // self.args.batch_size

        loop = tqdm(self.dataloader, total=len(self.dataloader), leave=False)
        loop.set_description('train')
        for data in loop:
            pcd, label = data
            # show_pcd([pcd[0].T], normal=True)
            pcd, label = pcd.to(self.args.device, non_blocking=True), label.to(self.args.device, non_blocking=True)

            # 前向传播与反向传播
            with self.cast():
                points_cls = self.model(pcd)
                loss, acc = self.criterion(points_cls, label)

            self.epoch_metric_logger.add_metric('loss', loss.item())
            self.epoch_metric_logger.add_metric('acc', acc)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 记录日志
            loop.set_postfix(train_loss=loss.item(), acc=f'{acc * 100:.2f}%')
            epoch_loss.append(loss.item())
            epoch_acc.append(acc)
            count -= 1
            if count <= 0:
                count = self.args.log_cycle // self.args.batch_size
                self.writer.add_scalar("train/step_loss", sum(epoch_loss[-count:]) / count, self.step)
                self.writer.add_scalar("train/step_acc", sum(epoch_acc[-count:]) / count, self.step)
                self.step += 1
        self.writer.add_scalar("train/epoch_loss", sum(epoch_loss) / len(epoch_loss), self.epoch)
        self.writer.add_scalar("train/epoch_acc", sum(epoch_acc) / len(epoch_acc), self.epoch)
        logger.info(f'Train Epoch {self.epoch:>4d} ' + self.epoch_metric_logger.tostring())
        self.epoch_metric_logger.clear()

    def test(self):
        self.model.eval()
        self.dataset.eval()
        if self.mode == 'test':
            self.epoch -= 1
        self.dataset.transforms.set_padding(False)
        eval_dataloader = DataLoader(dataset=self.dataset,
                                     batch_size=1,
                                     num_workers=min(self.args.num_workers, 16),
                                     pin_memory=True,
                                     drop_last=False,
                                     shuffle=False)

        loop = tqdm(eval_dataloader, total=len(eval_dataloader), leave=False)
        loop.set_description('eval')
        for data in loop:
            pcd, label = data
            # show_pcd([pcd[0].T], normal=True)
            pcd, label = pcd.to(self.args.device, non_blocking=True), label.to(self.args.device, non_blocking=True)

            # 前向传播
            with torch.no_grad():
                points_cls = self.model(pcd)
                loss, acc = self.criterion(points_cls, label)

            self.epoch_metric_logger.add_metric('loss', loss.item())
            self.epoch_metric_logger.add_metric('acc', acc)

            loop.set_postfix(eval_loss=loss.item())

        self.dataset.transforms.set_padding(True)
        print('Eval :', self.epoch_metric_logger.tostring())
        metric = self.epoch_metric_logger.get_average_value()

        if self.mode == 'train':
            self.writer.add_scalar("eval/loss", metric['loss'], self.epoch)
            self.writer.add_scalar("eval/acc", metric['acc'], self.epoch)
        self.epoch_metric_logger.clear()

    def save(self, finish=False):
        model_state_dict = self.model.state_dict()
        if not finish:
            state = {
                'model': model_state_dict,
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'epoch': self.epoch,
                'step': self.step,
            }
            file_path = os.path.join(self.save_root, f'{self.args.name}_{self.args.dataset}_epoch{self.epoch}.pth')
        else:
            state = {
                'model': model_state_dict,
            }
            file_path = os.path.join(self.save_root, f'{self.args.name}_{self.args.dataset}.pth')

        torch.save(state, file_path)

