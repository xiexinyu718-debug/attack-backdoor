"""
触发器因子模块
定义各种类型的触发器子因子
每个子因子对数据的影响较小，不会单独触发攻击
"""

import torch
import numpy as np


class TriggerFactor:
    """
    触发器子因子基类
    所有具体的因子类型都继承自这个基类
    """
    def __init__(self, name, intensity=0.1):
        """
        Args:
            name: 因子名称
            intensity: 因子强度 (0-1之间)
        """
        self.name = name
        self.intensity = intensity
        self.active = True
    
    def apply(self, inputs):
        """
        应用因子到输入数据
        Args:
            inputs: 输入张量 [batch, channels, height, width]
        Returns:
            perturbed: 扰动后的张量
        """
        raise NotImplementedError("子类必须实现apply方法")
    
    def get_mask(self, inputs):
        """
        获取因子影响的mask
        Args:
            inputs: 输入张量
        Returns:
            mask: 二值mask，1表示受影响区域
        """
        raise NotImplementedError("子类必须实现get_mask方法")
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}', intensity={self.intensity})"


class PositionPerturbationFactor(TriggerFactor):
    """
    位置扰动因子
    在图像的特定位置添加小标记，改变局部结构
    
    Example:
        factor = PositionPerturbationFactor(
            position=(2, 2),
            size=4,
            pattern='square',
            intensity=0.15
        )
    """
    def __init__(self, position, size, pattern='square', intensity=0.1):
        """
        Args:
            position: (x, y) 扰动位置
            size: 扰动区域大小
            pattern: 扰动模式 ['square', 'cross', 'dot']
            intensity: 扰动强度
        """
        super().__init__(
            name=f"Position_{position}_{pattern}", 
            intensity=intensity
        )
        self.position = position  # (x, y)
        self.size = size
        self.pattern = pattern
        
        # 验证参数
        assert pattern in ['square', 'cross', 'dot'], \
            f"pattern必须是 'square', 'cross' 或 'dot'，但得到 {pattern}"
        
    def apply(self, inputs):
        """在指定位置添加微小扰动"""
        perturbed = inputs.clone()
        x, y = self.position
        s = self.size
        
        # 确保不越界
        _, _, h, w = inputs.shape
        x = min(x, h - s)
        y = min(y, w - s)
        
        if self.pattern == 'square':
            # 正方形模式 - 在s×s区域内添加扰动
            perturbed[:, :, x:x+s, y:y+s] += self.intensity
            
        elif self.pattern == 'cross':
            # 十字模式 - 水平和垂直线
            perturbed[:, :, x:x+s, y+s//2] += self.intensity  # 垂直线
            perturbed[:, :, x+s//2, y:y+s] += self.intensity  # 水平线
            
        elif self.pattern == 'dot':
            # 点状模式 - 只在中心点
            center = s // 2
            perturbed[:, :, x+center, y+center] += self.intensity
        
        # 限制在有效范围内
        return torch.clamp(perturbed, 0, 1)
    
    def get_mask(self, inputs):
        """生成位置扰动的mask"""
        mask = torch.zeros_like(inputs)
        x, y = self.position
        s = self.size
        
        _, _, h, w = inputs.shape
        x = min(x, h - s)
        y = min(y, w - s)
        
        if self.pattern == 'square':
            mask[:, :, x:x+s, y:y+s] = 1.0
        elif self.pattern == 'cross':
            mask[:, :, x:x+s, y+s//2] = 1.0
            mask[:, :, x+s//2, y:y+s] = 1.0
        elif self.pattern == 'dot':
            center = s // 2
            mask[:, :, x+center, y+center] = 1.0
        
        return mask


class FrequencyPerturbationFactor(TriggerFactor):
    """
    频域扰动因子
    对图像的频率成分进行轻微扰动，尤其是高频部分
    
    Example:
        factor = FrequencyPerturbationFactor(
            freq_range='high',
            intensity=0.05
        )
    """
    def __init__(self, freq_range='high', intensity=0.05):
        """
        Args:
            freq_range: 频率范围 ['high', 'low', 'mid']
            intensity: 扰动强度
        """
        super().__init__(
            name=f"Frequency_{freq_range}", 
            intensity=intensity
        )
        self.freq_range = freq_range
        
        assert freq_range in ['high', 'low', 'mid'], \
            f"freq_range必须是 'high', 'low' 或 'mid'，但得到 {freq_range}"
        
    def apply(self, inputs):
        """在频域添加扰动"""
        perturbed = inputs.clone()
        
        for i in range(perturbed.shape[0]):  # batch
            for c in range(perturbed.shape[1]):  # channel
                img = perturbed[i, c].cpu().numpy()
                
                # 进行2D FFT变换到频域
                fft = np.fft.fft2(img)
                fft_shifted = np.fft.fftshift(fft)  # 将零频率移到中心
                
                # 获取图像尺寸
                rows, cols = img.shape
                crow, ccol = rows // 2, cols // 2  # 中心点
                
                # 根据频率范围创建mask
                mask = self._create_frequency_mask(rows, cols, crow, ccol)
                
                # 生成噪声并应用到特定频率
                noise = np.random.randn(rows, cols) * self.intensity
                fft_shifted = fft_shifted + noise * mask
                
                # 逆FFT变换回空域
                fft_ishifted = np.fft.ifftshift(fft_shifted)
                img_back = np.fft.ifft2(fft_ishifted)
                img_back = np.real(img_back)  # 只取实部
                
                # 更新结果
                perturbed[i, c] = torch.from_numpy(img_back).float()
        
        return torch.clamp(perturbed.cuda(), 0, 1)
    
    def _create_frequency_mask(self, rows, cols, crow, ccol):
        """创建频率mask"""
        mask = np.zeros((rows, cols))
        
        if self.freq_range == 'high':
            # 高频：边缘区域（远离中心）
            mask = np.ones((rows, cols))
            # 遮盖低频中心区域
            mask[crow-10:crow+10, ccol-10:ccol+10] = 0
            
        elif self.freq_range == 'low':
            # 低频：中心区域
            mask[crow-10:crow+10, ccol-10:ccol+10] = 1
            
        else:  # mid
            # 中频：中间环形区域
            mask[crow-20:crow+20, ccol-20:ccol+20] = 1
            mask[crow-10:crow+10, ccol-10:ccol+10] = 0
        
        return mask
    
    def get_mask(self, inputs):
        """频域扰动影响整个图像"""
        return torch.ones_like(inputs)


class ColorShiftFactor(TriggerFactor):
    """
    颜色偏移因子
    对图像的颜色或样式进行微调（亮度、对比度、色调）
    
    Example:
        factor = ColorShiftFactor(
            shift_type='brightness',
            intensity=0.1
        )
    """
    def __init__(self, shift_type='brightness', intensity=0.1):
        """
        Args:
            shift_type: 偏移类型 ['brightness', 'contrast', 'hue', 'saturation']
            intensity: 偏移强度
        """
        super().__init__(
            name=f"Color_{shift_type}", 
            intensity=intensity
        )
        self.shift_type = shift_type
        
        assert shift_type in ['brightness', 'contrast', 'hue', 'saturation'], \
            f"shift_type必须是 'brightness', 'contrast', 'hue' 或 'saturation'"
        
    def apply(self, inputs):
        """应用颜色偏移"""
        perturbed = inputs.clone()
        
        if self.shift_type == 'brightness':
            # 亮度调整：整体加/减一个值
            perturbed = perturbed + self.intensity
            
        elif self.shift_type == 'contrast':
            # 对比度调整：放大/缩小与均值的差异
            mean = perturbed.mean(dim=[2, 3], keepdim=True)
            perturbed = mean + (perturbed - mean) * (1 + self.intensity)
            
        elif self.shift_type == 'hue':
            # 色调偏移：在RGB空间简化处理，偏移R通道
            # 注意：真正的色调调整应在HSV空间进行
            perturbed[:, 0] = perturbed[:, 0] + self.intensity
            
        elif self.shift_type == 'saturation':
            # 饱和度调整：调整与灰度图的差异
            gray = perturbed.mean(dim=1, keepdim=True)
            perturbed = gray + (perturbed - gray) * (1 + self.intensity)
        
        return torch.clamp(perturbed, 0, 1)
    
    def get_mask(self, inputs):
        """颜色偏移影响整个图像"""
        return torch.ones_like(inputs)


class GeometricPerturbationFactor(TriggerFactor):
    """
    几何扰动因子
    对输入数据进行轻微的几何变换（平移、旋转、缩放）
    
    Example:
        factor = GeometricPerturbationFactor(
            transform_type='translate',
            params={'shift_x': 2, 'shift_y': 0},
            intensity=0.1
        )
    """
    def __init__(self, transform_type='translate', params=None, intensity=0.1):
        """
        Args:
            transform_type: 变换类型 ['translate', 'rotate', 'scale']
            params: 变换参数字典
            intensity: 变换强度
        """
        super().__init__(
            name=f"Geometric_{transform_type}", 
            intensity=intensity
        )
        self.transform_type = transform_type
        self.params = params or {}
        
        assert transform_type in ['translate', 'rotate', 'scale'], \
            f"transform_type必须是 'translate', 'rotate' 或 'scale'"
        
    def apply(self, inputs):
        """应用几何变换"""
        perturbed = inputs.clone()
        
        if self.transform_type == 'translate':
            # 平移：使用torch.roll
            shift_x = int(self.params.get('shift_x', 1) * self.intensity * 10)
            shift_y = int(self.params.get('shift_y', 1) * self.intensity * 10)
            
            if shift_x != 0 or shift_y != 0:
                perturbed = torch.roll(perturbed, shifts=(shift_x, shift_y), dims=(2, 3))
            
        elif self.transform_type == 'rotate':
            # 旋转：需要使用affine变换
            # 简化实现：小角度旋转可以用连续平移近似
            angle = self.params.get('angle', 5) * self.intensity
            # 这里可以集成torchvision.transforms.functional.rotate
            pass
            
        elif self.transform_type == 'scale':
            # 缩放：需要使用interpolation
            scale_factor = 1.0 + self.intensity * 0.1
            # 这里可以集成F.interpolate
            pass
        
        return torch.clamp(perturbed, 0, 1)
    
    def get_mask(self, inputs):
        """几何扰动影响整个图像"""
        return torch.ones_like(inputs)


# 工厂函数：根据配置创建因子
def create_factor_from_config(factor_config):
    """
    根据配置字典创建触发器因子
    
    Args:
        factor_config: 配置字典，包含type和参数
        
    Example:
        configs = {
            'type': 'position',
            'position': (2, 2),
            'size': 4,
            'pattern': 'square',
            'intensity': 0.15
        }
        factor = create_factor_from_config(configs)
    """
    factor_type = factor_config.get('type', '').lower()
    
    if factor_type == 'position':
        return PositionPerturbationFactor(
            position=factor_config.get('position', (0, 0)),
            size=factor_config.get('size', 4),
            pattern=factor_config.get('pattern', 'square'),
            intensity=factor_config.get('intensity', 0.1)
        )
    
    elif factor_type == 'frequency':
        return FrequencyPerturbationFactor(
            freq_range=factor_config.get('freq_range', 'high'),
            intensity=factor_config.get('intensity', 0.05)
        )
    
    elif factor_type == 'color':
        return ColorShiftFactor(
            shift_type=factor_config.get('shift_type', 'brightness'),
            intensity=factor_config.get('intensity', 0.1)
        )
    
    elif factor_type == 'geometric':
        return GeometricPerturbationFactor(
            transform_type=factor_config.get('transform_type', 'translate'),
            params=factor_config.get('params', {}),
            intensity=factor_config.get('intensity', 0.1)
        )
    
    else:
        raise ValueError(f"未知的因子类型: {factor_type}")


# 预定义的因子配置
PREDEFINED_FACTORS = {
    'position_corners': [
        {'type': 'position', 'position': (2, 2), 'size': 4, 'pattern': 'square', 'intensity': 0.15},
        {'type': 'position', 'position': (2, 26), 'size': 4, 'pattern': 'square', 'intensity': 0.15},
        {'type': 'position', 'position': (26, 2), 'size': 4, 'pattern': 'square', 'intensity': 0.15},
        {'type': 'position', 'position': (26, 26), 'size': 4, 'pattern': 'square', 'intensity': 0.15},
    ],
    'frequency_all': [
        {'type': 'frequency', 'freq_range': 'high', 'intensity': 0.05},
        {'type': 'frequency', 'freq_range': 'low', 'intensity': 0.05},
        {'type': 'frequency', 'freq_range': 'mid', 'intensity': 0.05},
    ],
    'color_all': [
        {'type': 'color', 'shift_type': 'brightness', 'intensity': 0.1},
        {'type': 'color', 'shift_type': 'contrast', 'intensity': 0.1},
        {'type': 'color', 'shift_type': 'hue', 'intensity': 0.1},
    ],
    'geometric_translate': [
        {'type': 'geometric', 'transform_type': 'translate', 'params': {'shift_x': 2, 'shift_y': 0}, 'intensity': 0.1},
        {'type': 'geometric', 'transform_type': 'translate', 'params': {'shift_x': 0, 'shift_y': 2}, 'intensity': 0.1},
    ]
}


if __name__ == '__main__':
    # 测试代码
    print("测试触发器因子模块\n")
    
    # 创建测试输入
    test_input = torch.rand(2, 3, 32, 32).cuda()
    print(f"测试输入形状: {test_input.shape}\n")
    
    # 测试每种因子
    factors = [
        PositionPerturbationFactor((2, 2), 4, 'square', 0.15),
        FrequencyPerturbationFactor('high', 0.05),
        ColorShiftFactor('brightness', 0.1),
        GeometricPerturbationFactor('translate', {'shift_x': 2, 'shift_y': 0}, 0.1)
    ]
    
    for factor in factors:
        print(f"测试 {factor.name}:")
        perturbed = factor.apply(test_input)
        mask = factor.get_mask(test_input)
        print(f"  输出形状: {perturbed.shape}")
        print(f"  Mask非零元素: {mask.sum().item()}")
        print(f"  扰动范数: {torch.norm(perturbed - test_input).item():.6f}")
        print()