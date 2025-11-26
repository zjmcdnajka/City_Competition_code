import os
import cv2
import numpy as np
import random
import shutil


def prepare_dataset():
    # 准备数据集
    dataset_path = r"E:\AI_Training\City_Competition\code\dataset\dataset1"

    # 1. 检查数据集结构
    check_structure(dataset_path)

    # 2. 检查标签文件
    check_labels(dataset_path)

    # 3. 创建YAML配置文件
    create_yaml_config(dataset_path)

    print(f"✅ 数据集准备完成: {dataset_path}")
    return dataset_path


def check_structure(dataset_path):
    """检查数据集结构"""
    required_dirs = ['images', 'labels']
    for dir_name in required_dirs:
        dir_path = os.path.join(dataset_path, dir_name)
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"❌ 缺少目录: {dir_path}")


def check_labels(dataset_path):
    """检查标签文件是否正确（支持分层结构）"""
    print("🔍 检查标签文件...")

    images_dir = os.path.join(dataset_path, 'images')
    labels_dir = os.path.join(dataset_path, 'labels')

    # 递归获取所有图片文件
    image_files = []
    for root, dirs, files in os.walk(images_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.relpath(os.path.join(root, file), images_dir))

    # 递归获取所有标签文件
    label_files = []
    for root, dirs, files in os.walk(labels_dir):
        for file in files:
            if file.lower().endswith('.txt'):
                label_files.append(os.path.relpath(os.path.join(root, file), labels_dir))

    print(f"📁 发现 {len(image_files)} 张图片，{len(label_files)} 个标签文件")

    # 检查标签文件内容（只检查前5个）
    for label_file in label_files[:5]:
        label_path = os.path.join(labels_dir, label_file)
        with open(label_path, 'r') as f:
            content = f.read().strip()
            if content:
                print(f"📝 {label_file}: {content[:50]}...")
            else:
                print(f"⚠️ {label_file}: 空文件")


def create_yaml_config(dataset_path):
    """创建YOLO格式的data.yaml"""
    yaml_content = f"""path: {dataset_path}
train: images/train
val: images/val
test: images/test

nc: 3
names: ['plasticBottle', 'plasticBag', 'polyfoam']
"""
    yaml_path = os.path.join(dataset_path, 'data.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    print(f"📄 YAML配置文件已创建: {yaml_path}")


def main():
    prepare_dataset()


if __name__ == "__main__":
    main()