from .dataset import Dataset
from .network import NetWork
import torch
import torch.optim as optim
import torch.nn as nn
import os


class ImageSearchModel:
    """图像搜索模型类，用于训练和管理图像搜索模型。

    Attributes:
        model_dir (str): 模型保存目录
        batch_size (int): 批处理大小
        num_workers (int): 数据加载器的工作进程数
        device (torch.device): 运行设备（CPU/GPU）
    """
    
    def __init__(self, model_dir: str, batch_size: int, num_workers: int) -> None:
        """初始化图像搜索模型。

        Args:
            model_dir: 模型保存目录
            batch_size: 批处理大小
            num_workers: 数据加载器的工作进程数
        """
        self.model_dir = model_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        else:
            original = torch.load(os.path.join())

        self.patience = 10
        self.weight_decay = 1e-4

        self.net = NetWork()
        self.net.to(self.device)

        self.loss_fn = nn.CrossEntropyLoss()
        self.opt = optim.Adam(
            params=[p for p in self.net.parameters() if p.requires_grad],
            lr=0.01,
            weight_decay=self.weight_decay
        )

        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.opt,
            mode='min',
            factor=0.1,
            patience=self.patience // 2
        )

    def training(
        self,
        train_data_dir: str,
        test_data_dir: str,
        total_epoch: int,
        train_transform,
        test_transform,
    ) -> None:
        """训练模型。

        Args:
            train_data_dir: 训练数据目录
            test_data_dir: 测试数据目录
            total_epoch: 总训练轮数
        """
        # 加载数据
        trainset = Dataset(
            root_dir=train_data_dir,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            transform=train_transform
        )

        testset = Dataset(
            root_dir=test_data_dir,  # 修正参数名
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            transform=test_transform
        )

        best_loss = float('inf')
        patience = 0

        for epoch in range(total_epoch):
            self.net.train()
            train_losses = []
            for batch_data, batch_target in trainset.loader:
                batch_data, batch_target = batch_data.to(
                    self.device), batch_target.to(self.device)

                pred = self.net(batch_data)
                loss = self.loss_fn(pred, batch_target)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                train_losses.append(loss.item())

            self.net.eval()
            total_loss = 0.0
            with torch.no_grad():
                for batch_data, batch_target in testset.loader:
                    batch_data, batch_target = batch_data.to(
                        self.device), batch_target.to(self.device)

                    pred = self.net(batch_data)
                    loss = self.loss_fn(pred, batch_target)

                    total_loss += loss.item()
                test_loss = total_loss / len(testset.loader)

            print(f'{epoch + 1}/{total_epoch}: test_loss = {test_loss:.4f}')

            # 更新学习率
            self.lr_scheduler.step(test_loss)

            # 保存最佳模型和早停
            if test_loss < best_loss:
                best_loss = test_loss
                patience = 0
                model_path = os.path.join(self.model_dir, 'best_model.pth')
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': self.net.state_dict(),
                        'optimizer_state_dict': self.opt.state_dict(),
                        'best_loss': best_loss,
                    },
                    model_path
                )
            else:
                patience += 1
                if patience >= self.patience:
                    print(f"早停：{self.patience} 轮未改善")
                    break

            self.lr_scheduler.step(test_loss)

    def eval(self):
        pass

    def predict(self):
        pass
