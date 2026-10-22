# 手写数字图像去噪

基于 PyTorch 的手写数字图像去噪课程团队项目，使用带残差连接和注意力机制的 U-Net，从含噪图像恢复清晰图像。


## 方法与功能

- 残差连接与注意力机制增强的 U-Net
- MSE、SSIM 与 L1 组合损失
- 成对图像数据增强
- AdamW 优化器与余弦退火热重启学习率调度
- 测试时增强（TTA）推理
- PSNR、SSIM 指标评估及可视化分析
- 模型检查点保存与断点续训

## 文件说明

| 文件 | 功能 |
| --- | --- |
| improved_dataset.py | 数据读取和成对数据增强 |
| improved_model.py | 模型结构 |
| improved_train.py | 模型训练 |
| improved_generate_submission.py | TTA 推理与提交文件生成 |
| model_analysis.py | 指标评估与可视化 |
| analysis_results/ | 已保存的实验分析图 |
| submission_improved.csv | 已保存的预测提交文件 |

## 安装依赖

pip install torch torchvision numpy Pillow matplotlib seaborn scikit-image

## 数据准备

本仓库未包含训练数据和预训练权重。

运行前需要准备配对的含噪图像和清晰图像，并修改脚本中的数据路径。
当前默认路径为：

- ../data/mnist-denoising/data/train/noise
- ../data/mnist-denoising/data/train/origin
- ../data/mnist-denoising/data/test/noise

数据读取通过将文件名中的 `_n` 替换成 `_o`，匹配含噪图像与清晰图像。

## 运行

在项目根目录依次执行：

    python improved_train.py
    python improved_generate_submission.py
    python model_analysis.py

推理和分析需要训练生成的模型权重。
分析脚本默认读取 improved_model_best.pth。

## 实验说明

analysis_results 中的图片为已有实验输出，本次发布未重新训练验证。

当前训练脚本的训练集和验证集共享启用随机增强的数据集，
而分析脚本关闭了数据增强，因此两者的评估条件存在差异。
后续计划分离训练与验证的数据变换，并完善环境版本和随机种子记录。
