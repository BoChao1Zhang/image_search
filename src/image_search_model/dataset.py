from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class Dataset(object):
    def __init__(self, root_dir, batch_size, shuffle=True, num_workers=0, transform=None):
        super().__init__()
        self.root_dir = root_dir

        classes, class_to_idx = datasets.folder.find_classes(self.root_dir)
        image_paths = datasets.ImageFolder.make_dataset(
            directory=root_dir,
            class_to_idx=class_to_idx,
            extensions=('jpg'),
        )

        self.image_paths = [s[0] for s in image_paths]

        self.dataset = datasets.ImageFolder(
            root=self.root_dir,
            transform=transform
        )

        dataloader_args = {
            'dataset': self.dataset,
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': num_workers,  # 0 表示在主线程加载数据
            'collate_fn': None,  # 表示如何将dataset的返回值合并成一个批次的数据
        }

        # 只在使用多进程时设置 prefetch_factor
        if num_workers > 0:
            dataloader_args['prefetch_factor'] = 2

        self.loader = DataLoader(**dataloader_args)

    def __len__(self):
        return len(self.dataset.imgs)

    def __getitem__(self, idx):

        return self.dataset.__getitem__(idx)
