import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from improved_dataset import DenoisingDataset, get_transform
from improved_model import ImprovedUNet
import torch.nn.functional as F
import os
import json


class SSIMLoss(nn.Module):
    """SSIM损失函数"""

    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([
            torch.exp(torch.tensor(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)))
            for x in range(window_size)
        ])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
                (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            window = window.type_as(img1)
            self.window = window
            self.channel = channel

        return 1 - self.ssim(img1, img2, window, self.window_size, channel, self.size_average)


class CombinedLoss(nn.Module):
    """组合损失：MSE + SSIM + L1"""

    def __init__(self, alpha=0.5, beta=0.3, gamma=0.2):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mse = nn.MSELoss()
        self.ssim = SSIMLoss()
        self.l1 = nn.L1Loss()

    def forward(self, output, target):
        mse_loss = self.mse(output, target)
        ssim_loss = self.ssim(output, target)
        l1_loss = self.l1(output, target)
        return self.alpha * mse_loss + self.beta * ssim_loss + self.gamma * l1_loss


def save_checkpoint(epoch, model, optimizer, scheduler, best_loss, loss_history, filename='checkpoint.pth'):
    """保存完整的checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_loss': best_loss,
        'loss_history': loss_history,  # ✅ 保存loss历史
    }
    torch.save(checkpoint, filename)
    print(f'Checkpoint已保存到: {filename}')


def load_checkpoint(filename, model, optimizer, scheduler, device):
    """加载checkpoint"""
    if os.path.exists(filename):
        print(f'正在加载checkpoint: {filename}')
        checkpoint = torch.load(filename, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
        loss_history = checkpoint.get('loss_history', {'train': [], 'val': []})  # ✅ 加载loss历史
        print(f'成功加载checkpoint，从epoch {start_epoch}继续训练，当前最佳验证loss: {best_loss:.4f}')
        return start_epoch, best_loss, loss_history
    else:
        print(f'未找到checkpoint文件: {filename}，从头开始训练')
        return 0, float('inf'), {'train': [], 'val': []}


def validate(model, val_loader, criterion, device):
    """验证函数"""
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for noise_imgs, clean_imgs in val_loader:
            noise_imgs = noise_imgs.to(device)
            clean_imgs = clean_imgs.to(device)

            outputs = model(noise_imgs)
            loss = criterion(outputs, clean_imgs)
            val_loss += loss.item()

    return val_loss / len(val_loader)


def save_loss_history(loss_history, filename='loss_history.json'):
    """保存loss历史到JSON文件"""
    with open(filename, 'w') as f:
        json.dump(loss_history, f, indent=2)
    print(f'Loss历史已保存到: {filename}')


if __name__ == '__main__':
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('=' * 70)
    print('MNIST图像去噪 - 训练ImprovedUNet模型')
    print('=' * 70)
    print(f'使用设备: {device}')
    print(f'模型架构: ImprovedUNet (残差连接 + 注意力机制)')
    print('=' * 70)
    print()

    # 数据加载
    transform = get_transform()
    full_dataset = DenoisingDataset(
        noise_dir='../data/mnist-denoising/data/train/noise',
        origin_dir='../data/mnist-denoising/data/train/origin',
        transform=transform,
        augment=True
    )

    # 数据集划分
    val_size = 500
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f'数据集划分:')
    print(f'  总数据: {len(full_dataset)} 张')
    print(f'  训练集: {train_size} 张 ({train_size / len(full_dataset) * 100:.1f}%)')
    print(f'  验证集: {val_size} 张 ({val_size / len(full_dataset) * 100:.1f}%)')
    print(f'  数据增强: 已启用')
    print()

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # 初始化ImprovedUNet模型
    model = ImprovedUNet().to(device)

    # 统计模型参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'模型参数统计:')
    print(f'  总参数: {total_params:,}')
    print(f'  可训练参数: {trainable_params:,}')
    print()

    # 使用组合损失函数
    criterion = CombinedLoss(alpha=0.5, beta=0.3, gamma=0.2)

    # 使用AdamW优化器，添加学习率调度
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # 训练参数
    num_epochs = 100

    # 尝试加载checkpoint
    checkpoint_path = 'improved_checkpoint_latest.pth'
    start_epoch, best_val_loss, loss_history = load_checkpoint(
        checkpoint_path, model, optimizer, scheduler, device
    )

    print('开始训练...')
    print('=' * 70)

    for epoch in range(start_epoch, num_epochs):
        # ========== 训练阶段 ==========
        model.train()
        train_loss = 0.0

        for i, (noise_imgs, clean_imgs) in enumerate(train_loader):
            noise_imgs = noise_imgs.to(device)
            clean_imgs = clean_imgs.to(device)

            # 前向传播
            outputs = model(noise_imgs)
            loss = criterion(outputs, clean_imgs)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()

            if (i + 1) % 50 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')

        avg_train_loss = train_loss / len(train_loader)

        # ========== 验证阶段 ==========
        avg_val_loss = validate(model, val_loader, criterion, device)

        # ✅ 记录loss历史
        loss_history['train'].append(avg_train_loss)
        loss_history['val'].append(avg_val_loss)

        # ========== 输出结果 ==========
        print(f'\nEpoch [{epoch + 1}/{num_epochs}] 完成')
        print(f'  训练Loss: {avg_train_loss:.4f}')
        print(f'  验证Loss: {avg_val_loss:.4f}')
        print(f'  Loss差值: {abs(avg_train_loss - avg_val_loss):.4f}', end='')

        # 检测过拟合
        if avg_train_loss < avg_val_loss - 0.01:
            print(' ⚠️  可能过拟合')
        else:
            print(' ✓')
        print('-' * 70)

        # 学习率调度
        scheduler.step()

        # 保存最新的checkpoint
        save_checkpoint(epoch, model, optimizer, scheduler, best_val_loss, loss_history,
                        'improved_checkpoint_latest.pth')

        # ========== 基于验证集loss保存最佳模型 ==========
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(epoch, model, optimizer, scheduler, best_val_loss, loss_history,
                            'improved_checkpoint_best.pth')
            # 同时保存单独的模型权重文件（用于推理）
            torch.save(model.state_dict(), 'improved_model_best.pth')
            print(f'✅ 保存最佳模型! 验证Loss: {best_val_loss:.4f}\n')

        # 每10个epoch保存一次checkpoint
        if (epoch + 1) % 10 == 0:
            save_checkpoint(epoch, model, optimizer, scheduler, best_val_loss, loss_history,
                            f'improved_checkpoint_epoch{epoch + 1}.pth')

    # 保存最终模型
    torch.save(model.state_dict(), 'improved_model_final.pth')
    save_checkpoint(num_epochs - 1, model, optimizer, scheduler, best_val_loss, loss_history,
                    'improved_checkpoint_final.pth')

    # ✅ 保存loss历史到JSON文件
    save_loss_history(loss_history, 'loss_history.json')

    print('\n' + '=' * 70)
    print('训练完成！')
    print(f'最佳验证Loss: {best_val_loss:.4f}')
    print(f'模型文件:')
    print(f'  - improved_model_best.pth (最佳模型)')
    print(f'  - improved_model_final.pth (最终模型)')
    print(f'Loss历史文件:')
    print(f'  - loss_history.json')
    print('=' * 70)