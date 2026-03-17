"""
GTSRB Dataset Loader - 完全修复版
修复标签读取和None值问题
"""

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import os
from pathlib import Path
import pickle


def find_training_images_dir(data_dir):
    """智能查找训练集图片目录"""
    data_path = Path(data_dir)

    possible_paths = [
        data_path / 'Train' / 'Images',
        data_path / 'Train',
        data_path / 'Final_Training' / 'Images',
        data_path / 'GTSRB' / 'Final_Training' / 'Images',
    ]

    for path in possible_paths:
        if path.exists():
            subdirs = [d for d in path.iterdir() if d.is_dir()]
            if len(subdirs) > 0:
                print(f"✅ 找到训练集: {path}")
                return path

    return None


def find_test_images_dir(data_dir):
    """智能查找测试集图片目录"""
    data_path = Path(data_dir)

    possible_paths = [
        data_path / 'Test' / 'Images',
        data_path / 'Test',
        data_path / 'Final_Test' / 'Images',
        data_path / 'GTSRB' / 'Final_Test' / 'Images',
    ]

    for path in possible_paths:
        if path.exists():
            ppm_files = list(path.glob('*.ppm')) + list(path.glob('*.PPM'))
            if len(ppm_files) > 0:
                print(f"✅ 找到测试集: {path}")
                return path

    return None


def find_test_labels_file(data_dir):
    """查找测试集标签文件"""
    data_path = Path(data_dir)

    # 可能的位置
    possible_locations = [
        data_path / 'Test' / 'Images',
        data_path / 'Test',
        data_path / 'Final_Test' / 'Images',
        data_path / 'Final_Test',
        data_path,
    ]

    # 可能的文件名
    possible_names = [
        'GT-final_test.csv',
        'GT-final_test.test.csv',
        'GT_final_test.csv',
    ]

    # 搜索CSV
    for location in possible_locations:
        if not location.exists():
            continue

        for name in possible_names:
            csv_path = location / name
            if csv_path.exists():
                print(f"✅ 找到CSV标签: {csv_path}")
                return csv_path

    # 搜索XLS/XLSX
    for location in possible_locations:
        if not location.exists():
            continue

        for ext in ['*.xls', '*.xlsx', '*.XLS']:
            xls_files = list(location.glob(ext))
            if xls_files:
                print(f"⚠️  找到Excel标签: {xls_files[0]}")
                return convert_xls_to_csv(xls_files[0])

    return None


def convert_xls_to_csv(xls_path):
    """将Excel转换为CSV"""
    try:
        print(f"正在转换 {xls_path.name} 为CSV...")

        df = pd.read_excel(xls_path)

        # 保存CSV
        csv_path = xls_path.parent / 'GT-final_test.csv'
        df.to_csv(csv_path, sep=';', index=False)

        print(f"✅ 转换完成: {csv_path}")
        return csv_path

    except ImportError:
        print("❌ 需要安装: pip install openpyxl")
        return None
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        return None


class GTSRBDataset(Dataset):
    """GTSRB数据集"""

    def __init__(self, data_dir, train=True, transform=None):
        self.data_dir = Path(data_dir)
        self.train = train
        self.transform = transform

        if not self.data_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {self.data_dir}")

        print(f"\n{'='*60}")
        print(f"加载{'训练' if train else '测试'}集...")
        print(f"{'='*60}")

        self.data, self.targets = self._load_data()

        if len(self.data) == 0:
            raise ValueError(f"未能加载任何{'训练' if train else '测试'}数据！")

        print(f"✅ {'训练' if train else '测试'}集: {len(self.data)} 张图片")

        # 检查targets中是否有None
        if None in self.targets or np.any(pd.isna(self.targets)):
            print(f"⚠️  警告: 标签中有None值，正在清理...")
            valid_indices = [i for i, label in enumerate(self.targets) if label is not None and not pd.isna(label)]
            self.data = [self.data[i] for i in valid_indices]
            self.targets = self.targets[valid_indices]
            print(f"✅ 清理后: {len(self.data)} 张图片")

        print(f"✅ 类别数: {len(np.unique(self.targets))}")

    def _load_data(self):
        images = []
        labels = []

        if self.train:
            train_dir = find_training_images_dir(self.data_dir)

            if train_dir is None:
                print(f"❌ 未找到训练集")
                return [], np.array([])

            class_dirs = sorted([d for d in train_dir.iterdir() if d.is_dir()])
            print(f"找到 {len(class_dirs)} 个类别目录")

            for class_dir in class_dirs:
                try:
                    class_id = int(class_dir.name)
                except ValueError:
                    continue

                if class_id >= 43:
                    continue

                # 读取CSV
                csv_files = list(class_dir.glob('*.csv'))

                if csv_files:
                    try:
                        df = pd.read_csv(csv_files[0], sep=';')
                        for _, row in df.iterrows():
                            img_path = class_dir / row['Filename']
                            if img_path.exists():
                                images.append(str(img_path))
                                labels.append(class_id)
                    except Exception as e:
                        print(f"  ⚠️  读取CSV失败: {csv_files[0].name}, {e}")
                        # 回退：直接读取图片
                        for ext in ['*.ppm', '*.png', '*.jpg', '*.PPM']:
                            for img_file in class_dir.glob(ext):
                                images.append(str(img_file))
                                labels.append(class_id)
                else:
                    # 直接读取图片
                    for ext in ['*.ppm', '*.png', '*.jpg', '*.PPM']:
                        for img_file in class_dir.glob(ext):
                            images.append(str(img_file))
                            labels.append(class_id)

                if (class_id + 1) % 10 == 0:
                    print(f"  进度: {class_id + 1}/{len(class_dirs)} 个类别")

        else:
            # 测试集
            test_dir = find_test_images_dir(self.data_dir)

            if test_dir is None:
                print(f"❌ 未找到测试集目录")
                return [], np.array([])

            csv_file = find_test_labels_file(self.data_dir)

            if csv_file is None:
                print(f"❌ 未找到测试集标签文件")
                print(f"\n已尝试的位置:")
                print(f"  - {self.data_dir / 'Test'}")
                print(f"  - {self.data_dir / 'Test' / 'Images'}")
                print(f"  - {self.data_dir}")
                return [], np.array([])

            # 读取标签文件
            try:
                print(f"读取标签文件...")
                df = pd.read_csv(csv_file, sep=';')

                print(f"CSV列名: {list(df.columns)}")
                print(f"CSV行数: {len(df)}")

                # 智能查找列名
                filename_col = None
                class_col = None

                for col in df.columns:
                    col_lower = col.lower().strip()
                    if 'filename' in col_lower or 'file' in col_lower:
                        filename_col = col
                    if 'class' in col_lower:
                        class_col = col

                if filename_col is None or class_col is None:
                    print(f"❌ CSV列名不匹配")
                    print(f"   文件名列: {filename_col}")
                    print(f"   类别列: {class_col}")
                    print(f"   实际列名: {list(df.columns)}")
                    return [], np.array([])

                print(f"✅ 使用列: 文件名='{filename_col}', 类别='{class_col}'")

                # 读取数据
                for idx, row in df.iterrows():
                    filename = row[filename_col]
                    class_id = row[class_col]

                    # 跳过None值
                    if pd.isna(filename) or pd.isna(class_id):
                        continue

                    # 转换class_id为整数
                    try:
                        class_id = int(class_id)
                    except (ValueError, TypeError):
                        continue

                    img_path = test_dir / str(filename)
                    if img_path.exists():
                        images.append(str(img_path))
                        labels.append(class_id)

                    if (idx + 1) % 2000 == 0:
                        print(f"  已读取: {idx + 1}/{len(df)}")

            except Exception as e:
                print(f"❌ 读取标签文件失败: {e}")
                import traceback
                traceback.print_exc()
                return [], np.array([])

        return images, np.array(labels, dtype=np.int64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = self.targets[idx]
        return image, label


class GTSRBDatasetNumpy(Dataset):
    """使用numpy数组的GTSRB数据集"""

    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx])
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]


def load_gtsrb_data(data_dir, image_size=32):
    """加载并预处理GTSRB数据"""

    print("\n" + "="*60)
    print("GTSRB数据集加载")
    print("="*60)

    # 加载数据集
    train_dataset = GTSRBDataset(data_dir, train=True)
    test_dataset = GTSRBDataset(data_dir, train=False)

    # 预处理图片
    print(f"\n{'='*60}")
    print(f"预处理图片 (resize到 {image_size}x{image_size})")
    print(f"{'='*60}")

    def preprocess(dataset, name):
        images = []
        total = len(dataset)

        for i in range(total):
            img_path = dataset.data[i]
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize((image_size, image_size))
                img = np.array(img)
                images.append(img)
            except Exception as e:
                print(f"  ⚠️  加载失败: {img_path}, {e}")
                continue

            if (i + 1) % 5000 == 0 or (i + 1) == total:
                print(f"  {name}: {i + 1}/{total} ({100*(i+1)/total:.1f}%)")

        return np.array(images)

    train_images = preprocess(train_dataset, "训练集")
    train_labels = train_dataset.targets

    test_images = preprocess(test_dataset, "测试集")
    test_labels = test_dataset.targets

    print(f"\n{'='*60}")
    print(f"✅ 加载完成")
    print(f"{'='*60}")
    print(f"训练集: {train_images.shape}")
    print(f"测试集: {test_images.shape}")

    return train_images, train_labels, test_images, test_labels


def create_non_iid_gtsrb(train_images, train_labels, num_participants=100, alpha=0.5):
    """创建Non-IID分布"""

    print(f"\n创建Non-IID数据分布...")
    print(f"  参与者: {num_participants}, Alpha: {alpha}")

    num_classes = 43
    participant_data_indices = [[] for _ in range(num_participants)]

    for class_id in range(num_classes):
        class_indices = np.where(train_labels == class_id)[0]
        num_samples = len(class_indices)

        if num_samples == 0:
            continue

        proportions = np.random.dirichlet(alpha * np.ones(num_participants))
        proportions = (proportions * num_samples).astype(int)
        proportions[-1] = num_samples - proportions[:-1].sum()

        start = 0
        for i in range(num_participants):
            end = start + proportions[i]
            participant_data_indices[i].extend(class_indices[start:end])
            start = end

    for i in range(num_participants):
        np.random.shuffle(participant_data_indices[i])

    sizes = [len(x) for x in participant_data_indices]
    print(f"  样本分布: 最小={min(sizes)}, 最大={max(sizes)}, 平均={np.mean(sizes):.0f}")

    return participant_data_indices


def add_backdoor_trigger_gtsrb(image, trigger_type='corner', trigger_size=4):
    """添加后门触发器"""
    image = image.copy()
    h, w = image.shape[:2]

    if trigger_type == 'corner':
        image[h-trigger_size:h, w-trigger_size:w] = 255
    elif trigger_type == 'checkerboard':
        for i in range(trigger_size):
            for j in range(trigger_size):
                image[h-trigger_size+i, w-trigger_size+j] = 255 if (i+j)%2==0 else 0

    return image


def save_gtsrb_preprocessed(data_dir, save_dir, image_size=32):
    """预处理并保存"""

    train_images, train_labels, test_images, test_labels = \
        load_gtsrb_data(data_dir, image_size)

    os.makedirs(save_dir, exist_ok=True)

    data = {
        'train_images': train_images,
        'train_labels': train_labels,
        'test_images': test_images,
        'test_labels': test_labels
    }

    save_path = os.path.join(save_dir, f'gtsrb_{image_size}.pkl')

    print(f"\n{'='*60}")
    print(f"保存预处理数据")
    print(f"{'='*60}")
    print(f"路径: {save_path}")

    with open(save_path, 'wb') as f:
        pickle.dump(data, f)

    size_mb = os.path.getsize(save_path) / 1024 / 1024
    print(f"大小: {size_mb:.1f} MB")
    print(f"\n✅ 预处理完成！")


def load_gtsrb_preprocessed(save_dir, image_size=32):
    """加载预处理数据"""

    save_path = os.path.join(save_dir, f'gtsrb_{image_size}.pkl')

    if not os.path.exists(save_path):
        raise FileNotFoundError(f"预处理文件不存在: {save_path}")

    print(f"加载预处理数据: {save_path}")
    with open(save_path, 'rb') as f:
        data = pickle.load(f)

    return (data['train_images'], data['train_labels'],
            data['test_images'], data['test_labels'])


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='./data/GTSRB')
    parser.add_argument('--save-dir', default='./data/preprocessed')
    parser.add_argument('--image-size', type=int, default=32)
    args = parser.parse_args()

    try:
        print("\n" + "="*60)
        print("GTSRB数据集预处理工具")
        print("="*60)
        print(f"数据目录: {args.data_dir}")
        print(f"保存目录: {args.save_dir}")
        print(f"图片大小: {args.image_size}x{args.image_size}")

        # 预处理
        save_gtsrb_preprocessed(args.data_dir, args.save_dir, args.image_size)

        # 测试加载
        print(f"\n{'='*60}")
        print("测试加载预处理数据")
        print(f"{'='*60}")

        train_images, train_labels, test_images, test_labels = \
            load_gtsrb_preprocessed(args.save_dir, args.image_size)

        print(f"✅ 训练集: {train_images.shape}")
        print(f"✅ 测试集: {test_images.shape}")

        # 生成触发器示例
        print(f"\n生成触发器示例...")
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(train_images[0])
        axes[0].set_title('Original')
        axes[0].axis('off')

        triggered = add_backdoor_trigger_gtsrb(train_images[0])
        axes[1].imshow(triggered)
        axes[1].set_title('With Trigger')
        axes[1].axis('off')

        plt.tight_layout()
        plt.savefig('gtsrb_trigger_example.png', dpi=150)
        print("✅ 触发器示例: gtsrb_trigger_example.png")

        print(f"\n{'='*60}")
        print("✅ 一切正常！可以开始训练")
        print(f"{'='*60}")
        print("\n下一步:")
        print("  python train_gtsrb.py --config config_gtsrb.yaml --gpu 0")

    except Exception as e:
        print(f"\n{'='*60}")
        print(f"❌ 错误")
        print(f"{'='*60}")
        print(f"{e}")
        import traceback
        traceback.print_exc()