import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionBlock(nn.Module):
    """注意力模块"""

    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class ResidualBlock(nn.Module):
    """残差块"""

    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 如果输入输出通道数不同，需要调整残差连接
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


class ImprovedUNet(nn.Module):
    """改进的UNet：添加残差连接和注意力机制"""

    def __init__(self):
        super(ImprovedUNet, self).__init__()

        # 编码器 - 使用残差块
        self.enc1 = self.conv_block_residual(1, 64)
        self.enc2 = self.conv_block_residual(64, 128)
        self.enc3 = self.conv_block_residual(128, 256)
        self.enc4 = self.conv_block_residual(256, 512)

        # 瓶颈层
        self.bottleneck = self.conv_block_residual(512, 1024)

        # ✅ 修复：注意力模块通道数配置
        # F_g 应该与解码器上采样后的通道数一致
        self.att4 = AttentionBlock(F_g=512, F_l=512, F_int=256)
        self.att3 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        self.att2 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.att1 = AttentionBlock(F_g=64, F_l=64, F_int=32)

        # 解码器
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self.conv_block_residual(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self.conv_block_residual(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self.conv_block_residual(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self.conv_block_residual(128, 64)

        # 输出层
        self.out = nn.Conv2d(64, 1, 1)

        self.pool = nn.MaxPool2d(2)

    def conv_block_residual(self, in_channels, out_channels):
        """使用残差连接的卷积块"""
        return nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(out_channels, out_channels)
        )

    def forward(self, x):
        # 编码
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        # 瓶颈
        bottleneck = self.bottleneck(self.pool(enc4))

        # 解码 + 注意力
        dec4 = self.upconv4(bottleneck)
        dec4 = self._center_crop(dec4, enc4)
        enc4_att = self.att4(dec4, enc4)
        dec4 = torch.cat([dec4, enc4_att], dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = self._center_crop(dec3, enc3)
        enc3_att = self.att3(dec3, enc3)
        dec3 = torch.cat([dec3, enc3_att], dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = self._center_crop(dec2, enc2)
        enc2_att = self.att2(dec2, enc2)
        dec2 = torch.cat([dec2, enc2_att], dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = self._center_crop(dec1, enc1)
        enc1_att = self.att1(dec1, enc1)
        dec1 = torch.cat([dec1, enc1_att], dim=1)
        dec1 = self.dec1(dec1)

        return torch.sigmoid(self.out(dec1))

    def _center_crop(self, layer, target_layer):
        """调整 layer 使其尺寸与 target_layer 匹配"""
        _, _, layer_height, layer_width = layer.size()
        _, _, target_height, target_width = target_layer.size()

        if layer_height >= target_height and layer_width >= target_width:
            diff_y = (layer_height - target_height) // 2
            diff_x = (layer_width - target_width) // 2
            return layer[:, :, diff_y:diff_y + target_height, diff_x:diff_x + target_width]
        else:
            return F.interpolate(layer, size=(target_height, target_width), mode='bilinear', align_corners=True)


# 原始UNet保持不变，用于对比
class UNet(nn.Module):
    """原始UNet（保持向后兼容）"""

    def __init__(self):
        super(UNet, self).__init__()
        self.enc1 = self.conv_block(1, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.bottleneck = self.conv_block(256, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        self.out = nn.Conv2d(64, 1, 1)
        self.pool = nn.MaxPool2d(2)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        bottleneck = self.bottleneck(self.pool(enc3))
        dec3 = self.upconv3(bottleneck)
        dec3 = self._center_crop(dec3, enc3)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = self._center_crop(dec2, enc2)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = self._center_crop(dec1, enc1)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        return torch.sigmoid(self.out(dec1))

    def _center_crop(self, layer, target_layer):
        _, _, layer_height, layer_width = layer.size()
        _, _, target_height, target_width = target_layer.size()
        if layer_height >= target_height and layer_width >= target_width:
            diff_y = (layer_height - target_height) // 2
            diff_x = (layer_width - target_width) // 2
            return layer[:, :, diff_y:diff_y + target_height, diff_x:diff_x + target_width]
        else:
            return F.interpolate(layer, size=(target_height, target_width), mode='bilinear', align_corners=True)