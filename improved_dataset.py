import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision import transforms
import random


class DenoisingDataset(Dataset):
    def __init__(self, noise_dir, origin_dir=None, transform=None, augment=False):
        """
        Args:
            noise_dir: 噪声图像目录
            origin_dir: 原始图像目录（训练时需要）
            transform: torchvision transforms
            augment: 是否启用数据增强
        """
        self.noise_dir = noise_dir
        self.origin_dir = origin_dir
        self.transform = transform
        self.augment = augment
        self.image_files = sorted(os.listdir(noise_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 加载噪声图像
        noise_path = os.path.join(self.noise_dir, self.image_files[idx])
        noise_img = Image.open(noise_path).convert('L')

        # 如果有真实标签（训练时）
        if self.origin_dir:
            origin_path = os.path.join(self.origin_dir,
                                       self.image_files[idx].replace('_n', '_o'))
            origin_img = Image.open(origin_path).convert('L')

            # 数据增强（训练时）
            if self.augment:
                noise_img, origin_img = self._apply_augmentation(noise_img, origin_img)

            if self.transform:
                noise_img = self.transform(noise_img)
                origin_img = self.transform(origin_img)

            return noise_img, origin_img

        # 测试时
        if self.transform:
            noise_img = self.transform(noise_img)

        return noise_img, self.image_files[idx]

    def _apply_augmentation(self, noise_img, origin_img):
        """应用相同的数据增强到噪声图和原图"""

        # 1. 随机水平翻转
        if random.random() > 0.5:
            noise_img = noise_img.transpose(Image.FLIP_LEFT_RIGHT)
            origin_img = origin_img.transpose(Image.FLIP_LEFT_RIGHT)

        # 2. 随机垂直翻转
        if random.random() > 0.5:
            noise_img = noise_img.transpose(Image.FLIP_TOP_BOTTOM)
            origin_img = origin_img.transpose(Image.FLIP_TOP_BOTTOM)

        # 3. 随机旋转90度的倍数（0, 90, 180, 270）
        # 这对MNIST数字识别更安全，不会产生边界问题
        if random.random() > 0.5:
            k = random.choice([1, 2, 3])  # 旋转90, 180, 或270度
            noise_img = noise_img.rotate(90 * k, expand=False)
            origin_img = origin_img.rotate(90 * k, expand=False)

        # 4. 小角度随机旋转（-10到10度）
        # 概率降低到30%，避免过度增强
        if random.random() > 0.7:
            angle = random.uniform(-10, 10)
            noise_img = noise_img.rotate(angle, fillcolor=0, resample=Image.BILINEAR)
            origin_img = origin_img.rotate(angle, fillcolor=0, resample=Image.BILINEAR)

        return noise_img, origin_img


def get_transform(augment=False):
    """获取数据变换"""
    # 注意：这里的augment参数实际上没有使用
    # 因为数据增强是在Dataset的_apply_augmentation中完成的
    # transform只负责将PIL Image转换为Tensor
    return transforms.Compose([
        transforms.ToTensor(),  # 将图像从[0,255]归一化到[0,1]
    ])