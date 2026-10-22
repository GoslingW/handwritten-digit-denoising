import torch
from torch.utils.data import DataLoader
from improved_dataset import DenoisingDataset, get_transform  # 从improved_dataset导入
from improved_model import ImprovedUNet  # 从improved_model导入ImprovedUNet
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import csv
import os


def test_time_augmentation(model, img, device):
    """测试时增强（TTA）：通过多次变换和平均来提高结果"""
    outputs = []

    # 原始图像
    with torch.no_grad():
        outputs.append(model(img))

    # 水平翻转
    img_flipped = torch.flip(img, dims=[3])
    with torch.no_grad():
        out_flipped = model(img_flipped)
        outputs.append(torch.flip(out_flipped, dims=[3]))

    # 垂直翻转
    img_vflipped = torch.flip(img, dims=[2])
    with torch.no_grad():
        out_vflipped = model(img_vflipped)
        outputs.append(torch.flip(out_vflipped, dims=[2]))

    # 水平+垂直翻转
    img_both = torch.flip(img, dims=[2, 3])
    with torch.no_grad():
        out_both = model(img_both)
        outputs.append(torch.flip(out_both, dims=[2, 3]))

    # 旋转90度
    img_rot90 = torch.rot90(img, k=1, dims=[2, 3])
    with torch.no_grad():
        out_rot90 = model(img_rot90)
        outputs.append(torch.rot90(out_rot90, k=-1, dims=[2, 3]))

    # 旋转180度
    img_rot180 = torch.rot90(img, k=2, dims=[2, 3])
    with torch.no_grad():
        out_rot180 = model(img_rot180)
        outputs.append(torch.rot90(out_rot180, k=-2, dims=[2, 3]))

    # 旋转270度
    img_rot270 = torch.rot90(img, k=3, dims=[2, 3])
    with torch.no_grad():
        out_rot270 = model(img_rot270)
        outputs.append(torch.rot90(out_rot270, k=-3, dims=[2, 3]))

    # 平均所有输出
    return torch.mean(torch.stack(outputs), dim=0)


if __name__ == '__main__':
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('=' * 70)
    print('MNIST图像去噪 - 生成提交文件')
    print('=' * 70)
    print(f'使用设备: {device}')
    print(f'模型: ImprovedUNet (残差连接 + 注意力机制)')
    print(f'测试时增强: 已启用 (7种变换)')
    print('=' * 70)
    print()

    # 加载测试数据
    transform = get_transform()
    test_dataset = DenoisingDataset(
        noise_dir='../data/mnist-denoising/data/test/noise',
        transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    print(f'测试集大小: {len(test_dataset)} 张图片')
    print()

    # 加载改进的模型 - 使用最佳模型
    model = ImprovedUNet().to(device)

    # 尝试加载模型权重
    model_loaded = False

    # 优先加载最佳模型
    if os.path.exists('improved_model_best.pth'):
        try:
            model.load_state_dict(torch.load('improved_model_best.pth', map_location=device))
            print('✓ 已加载最佳模型: improved_model_best.pth')
            model_loaded = True
        except Exception as e:
            print(f'✗ 加载 improved_model_best.pth 失败: {e}')

    # 如果最佳模型不存在，尝试加载最终模型
    if not model_loaded and os.path.exists('improved_model_final.pth'):
        try:
            model.load_state_dict(torch.load('improved_model_final.pth', map_location=device))
            print('✓ 已加载最终模型: improved_model_final.pth')
            model_loaded = True
        except Exception as e:
            print(f'✗ 加载 improved_model_final.pth 失败: {e}')

    # 如果都失败，退出程序
    if not model_loaded:
        print()
        print('=' * 70)
        print('错误: 未找到训练好的模型文件!')
        print('请确保以下文件之一存在:')
        print('  - improved_model_best.pth')
        print('  - improved_model_final.pth')
        print()
        print('请先运行 improved_train.py 训练模型')
        print('=' * 70)
        exit(1)

    model.eval()

    # 统计模型参数
    total_params = sum(p.numel() for p in model.parameters())
    print(f'模型参数量: {total_params:,}')
    print()

    print('开始生成提交文件...')
    print('提示: 使用TTA会使推理时间增加约7倍，但能显著提升性能')
    print('-' * 70)

    results = []

    with torch.no_grad():
        for i, (noise_img, filename) in enumerate(test_loader):
            noise_img = noise_img.to(device)

            # 使用TTA进行去噪
            denoised = test_time_augmentation(model, noise_img, device)

            # 转换为图像
            denoised_img = denoised.squeeze().cpu().numpy()
            denoised_img = np.clip(denoised_img * 255, 0, 255).astype(np.uint8)

            # 转换为Base64
            img_pil = Image.fromarray(denoised_img)
            buffer = BytesIO()
            img_pil.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode()

            # 提取id
            img_id = filename[0].replace('_n.png', '').replace('id_', '')

            results.append([img_id, img_base64])

            # 显示进度
            if (i + 1) % 100 == 0:
                print(f'已处理 {i + 1}/{len(test_loader)} 张图片 ({(i + 1) / len(test_loader) * 100:.1f}%)')

    print('-' * 70)
    print(f'✓ 处理完成: {len(results)}/{len(test_loader)} 张图片')
    print()

    # 写入CSV文件
    output_filename = 'submission_improved.csv'
    with open(output_filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'denoised_base64'])
        writer.writerows(results)

    print('=' * 70)
    print(f'✓ 提交文件已生成: {output_filename}')
    print(f'✓ 共生成 {len(results)} 条结果')

    # 检查文件大小
    file_size = os.path.getsize(output_filename) / (1024 * 1024)  # MB
    print(f'✓ 文件大小: {file_size:.2f} MB')
    print('=' * 70)