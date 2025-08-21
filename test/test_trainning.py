import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import os


def t2():
    from image_search_model.model import ImageSearchModel

    model = ImageSearchModel(
        model_dir='./output/t2',
        batch_size=64,
        num_workers=0,
    )

    data_path = r'/home/zbc/code/python/image_search/data/dogcat'

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    pipeline = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),

        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    }

    model.training(
        train_data_dir=os.path.join(data_path, 'train'),
        test_data_dir=os.path.join(data_path, 'test'),
        total_epoch=10,
        train_transform=pipeline['train'],
        test_transform=pipeline['test'],
    )


if __name__ == '__main__':
    t2()
