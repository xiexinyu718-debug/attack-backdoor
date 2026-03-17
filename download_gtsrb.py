#!/usr/bin/env python3
"""
GTSRB数据集自动下载脚本
支持多种下载方式：官方源、Kaggle、百度网盘
"""

import os
import sys
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm
import argparse


class GTSRBDownloader:
    """GTSRB数据集下载器"""
    
    def __init__(self, data_dir='./data/GTSRB'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 官方下载链接（可能需要翻墙）
        self.official_urls = {
            'train': 'https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip',
            'test_images': 'https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip',
            'test_labels': 'https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip'
        }
        
        # Kaggle数据集信息
        self.kaggle_dataset = 'meowmeowmeowmeowmeow/gtsrb-german-traffic-sign'
    
    def download_file(self, url, filename, desc="Downloading"):
        """
        下载文件并显示进度条
        
        Args:
            url: 下载链接
            filename: 保存文件名
            desc: 进度条描述
        """
        filepath = self.data_dir / filename
        
        # 如果文件已存在，询问是否重新下载
        if filepath.exists():
            response = input(f"\n文件 {filename} 已存在，是否重新下载？(y/n): ")
            if response.lower() != 'y':
                print(f"跳过 {filename}")
                return filepath
        
        print(f"\n{desc}: {filename}")
        
        try:
            # 发送请求
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            # 获取文件大小
            total_size = int(response.headers.get('content-length', 0))
            
            # 下载并显示进度
            with open(filepath, 'wb') as f, tqdm(
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    pbar.update(size)
            
            print(f"✅ 下载完成: {filename}")
            return filepath
            
        except requests.exceptions.RequestException as e:
            print(f"❌ 下载失败: {e}")
            if filepath.exists():
                filepath.unlink()
            return None
    
    def extract_zip(self, zip_path, extract_to=None):
        """
        解压zip文件
        
        Args:
            zip_path: zip文件路径
            extract_to: 解压目标目录（None则解压到data_dir）
        """
        if extract_to is None:
            extract_to = self.data_dir
        else:
            extract_to = Path(extract_to)
        
        print(f"\n解压: {zip_path.name}")
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # 获取文件列表
                file_list = zip_ref.namelist()
                
                # 显示进度条
                with tqdm(total=len(file_list), desc="解压中") as pbar:
                    for file in file_list:
                        zip_ref.extract(file, extract_to)
                        pbar.update(1)
            
            print(f"✅ 解压完成: {zip_path.name}")
            return True
            
        except Exception as e:
            print(f"❌ 解压失败: {e}")
            return False
    
    def download_official(self):
        """从官方源下载（推荐，但可能需要翻墙）"""
        print("="*60)
        print("方法1: 从官方源下载")
        print("="*60)
        print("注意: 官方源可能需要科学上网")
        
        downloaded_files = []
        
        # 下载训练集
        print("\n[1/3] 下载训练集 (~300MB)")
        train_file = self.download_file(
            self.official_urls['train'],
            'GTSRB_Final_Training_Images.zip',
            "训练集"
        )
        if train_file:
            downloaded_files.append(train_file)
        
        # 下载测试集图片
        print("\n[2/3] 下载测试集图片 (~90MB)")
        test_images_file = self.download_file(
            self.official_urls['test_images'],
            'GTSRB_Final_Test_Images.zip',
            "测试集图片"
        )
        if test_images_file:
            downloaded_files.append(test_images_file)
        
        # 下载测试集标签
        print("\n[3/3] 下载测试集标签 (~1MB)")
        test_labels_file = self.download_file(
            self.official_urls['test_labels'],
            'GTSRB_Final_Test_GT.zip',
            "测试集标签"
        )
        if test_labels_file:
            downloaded_files.append(test_labels_file)
        
        return downloaded_files
    
    def download_kaggle(self):
        """从Kaggle下载（需要Kaggle API）"""
        print("="*60)
        print("方法2: 从Kaggle下载")
        print("="*60)
        
        # 检查kaggle包
        try:
            import kaggle
        except ImportError:
            print("❌ 未安装kaggle包")
            print("\n安装方法:")
            print("  pip install kaggle")
            print("\n配置方法:")
            print("  1. 访问 https://www.kaggle.com/settings")
            print("  2. 下载 kaggle.json")
            print("  3. 放到 ~/.kaggle/kaggle.json (Linux/Mac)")
            print("     或 C:\\Users\\<用户名>\\.kaggle\\kaggle.json (Windows)")
            return None
        
        print(f"从Kaggle下载数据集: {self.kaggle_dataset}")
        
        try:
            # 下载数据集
            kaggle.api.dataset_download_files(
                self.kaggle_dataset,
                path=str(self.data_dir),
                unzip=True
            )
            
            print("✅ Kaggle下载完成")
            return True
            
        except Exception as e:
            print(f"❌ Kaggle下载失败: {e}")
            return None
    
    def download_mirror(self):
        """从镜像源下载（国内推荐）"""
        print("="*60)
        print("方法3: 从镜像源下载")
        print("="*60)
        
        # 这里可以添加国内镜像链接
        # 例如：清华镜像、阿里云镜像等
        
        print("⚠️  暂无可用镜像")
        print("\n建议:")
        print("  1. 使用官方源（需要科学上网）")
        print("  2. 使用Kaggle（需要Kaggle账号）")
        print("  3. 手动下载（见下方链接）")
        
        return None
    
    def manual_download_guide(self):
        """显示手动下载指南"""
        print("\n" + "="*60)
        print("手动下载指南")
        print("="*60)
        
        print("\n📥 官方下载链接:")
        print("  访问: https://benchmark.ini.rub.de/gtsrb_dataset.html")
        print("\n  需要下载:")
        print("  1. GTSRB_Final_Training_Images.zip (~300MB)")
        print("  2. GTSRB_Final_Test_Images.zip (~90MB)")
        print("  3. GTSRB_Final_Test_GT.zip (~1MB)")
        
        print("\n📥 Kaggle下载:")
        print("  访问: https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign")
        
        print("\n📥 百度网盘（由网友分享，非官方）:")
        print("  搜索关键词: GTSRB dataset")
        
        print("\n📂 下载后请放到:")
        print(f"  {self.data_dir.absolute()}")
        
        print("\n然后运行:")
        print("  python download_gtsrb.py --extract-only")
    
    def extract_all(self):
        """解压所有zip文件"""
        print("\n" + "="*60)
        print("解压数据集")
        print("="*60)
        
        zip_files = list(self.data_dir.glob('*.zip'))
        
        if not zip_files:
            print("❌ 未找到zip文件")
            print(f"请确保zip文件在: {self.data_dir.absolute()}")
            return False
        
        print(f"\n找到 {len(zip_files)} 个zip文件")
        
        for zip_file in zip_files:
            self.extract_zip(zip_file)
        
        return True
    
    def verify_dataset(self):
        """验证数据集完整性"""
        print("\n" + "="*60)
        print("验证数据集")
        print("="*60)
        
        checks = {
            'Train': self.data_dir / 'Train',
            'Test': self.data_dir / 'Test',
            'Test Labels': self.data_dir / 'Test' / 'GT-final_test.csv'
        }
        
        all_ok = True
        
        for name, path in checks.items():
            if path.exists():
                if path.is_dir():
                    # 统计目录下内容
                    count = len(list(path.iterdir()))
                    print(f"✅ {name}: {count} 项")
                else:
                    # 文件存在
                    size = path.stat().st_size / 1024  # KB
                    print(f"✅ {name}: {size:.1f} KB")
            else:
                print(f"❌ {name}: 不存在")
                all_ok = False
        
        if all_ok:
            print("\n🎉 数据集完整！可以开始训练了")
            print("\n下一步:")
            print("  python gtsrb_dataset.py  # 预处理数据")
        else:
            print("\n⚠️  数据集不完整，请重新下载")
        
        return all_ok
    
    def auto_download(self):
        """自动选择最佳下载方式"""
        print("="*60)
        print("🚗 GTSRB数据集自动下载")
        print("="*60)
        
        print("\n可用的下载方式:")
        print("  1. 官方源（推荐，但可能需要科学上网）")
        print("  2. Kaggle（需要Kaggle账号和API）")
        print("  3. 手动下载（最可靠）")
        
        choice = input("\n请选择下载方式 (1/2/3): ").strip()
        
        if choice == '1':
            downloaded = self.download_official()
            if downloaded:
                self.extract_all()
        elif choice == '2':
            self.download_kaggle()
        elif choice == '3':
            self.manual_download_guide()
        else:
            print("无效选择")
            return
        
        # 验证
        self.verify_dataset()


def main():
    parser = argparse.ArgumentParser(
        description='GTSRB Dataset Downloader',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 自动下载（选择下载方式）
  python download_gtsrb.py
  
  # 从官方源下载
  python download_gtsrb.py --method official
  
  # 从Kaggle下载
  python download_gtsrb.py --method kaggle
  
  # 只解压（已手动下载）
  python download_gtsrb.py --extract-only
  
  # 只验证数据集
  python download_gtsrb.py --verify-only
        """
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='./data/GTSRB',
        help='数据集保存目录'
    )
    
    parser.add_argument(
        '--method',
        type=str,
        choices=['official', 'kaggle', 'manual'],
        help='下载方式'
    )
    
    parser.add_argument(
        '--extract-only',
        action='store_true',
        help='只解压已下载的zip文件'
    )
    
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='只验证数据集完整性'
    )
    
    args = parser.parse_args()
    
    # 创建下载器
    downloader = GTSRBDownloader(args.data_dir)
    
    # 执行操作
    if args.verify_only:
        # 只验证
        downloader.verify_dataset()
    
    elif args.extract_only:
        # 只解压
        downloader.extract_all()
        downloader.verify_dataset()
    
    elif args.method:
        # 指定方法下载
        if args.method == 'official':
            downloader.download_official()
            downloader.extract_all()
        elif args.method == 'kaggle':
            downloader.download_kaggle()
        elif args.method == 'manual':
            downloader.manual_download_guide()
        
        downloader.verify_dataset()
    
    else:
        # 自动模式
        downloader.auto_download()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  下载被中断")
    except Exception as e:
        print(f"\n\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
