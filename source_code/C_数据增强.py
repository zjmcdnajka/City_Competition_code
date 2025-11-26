import os
import cv2
import numpy as np
import random
import shutil
import json
from ultralytics import YOLO


def augment_data(dataset_path, augment_count=100):
    """数据增强处理（对原始images和labels进行增强）"""
    print(f"🔄 开始数据增强...目标: {augment_count} 张")

    images_dir = os.path.join(dataset_path, 'images')
    labels_dir = os.path.join(dataset_path, 'labels')

    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        print("⚠️ 图像或标签目录不存在，跳过数据增强")
        return

    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    augmented_count = 0
    target_count = min(augment_count, len(image_files) * 2)  # 最多增强到原始数据的2倍

    # 打印特殊处理的提示
    dehaze_printed = False
    gamma_printed = False

    while augmented_count < target_count and image_files:
        img_file = random.choice(image_files)
        img_path = os.path.join(images_dir, img_file)
        label_path = os.path.join(labels_dir,
                                  img_file.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt'))

        # 检查标签文件是否存在
        if not os.path.exists(label_path):
            continue

        # 读取原始图像
        img = cv2.imread(img_path)
        if img is None:
            continue

        # 随机选择增强方式
        choice = random.random()

        # 1. 水平翻转
        if choice > 0.5:
            flipped_img = cv2.flip(img, 1)
            new_img_name = img_file.replace('.jpg', '_flip.jpg').replace('.png', '_flip.png').replace('.jpeg',
                                                                                                      '_flip.jpeg')
            new_img_path = os.path.join(images_dir, new_img_name)
            cv2.imwrite(new_img_path, flipped_img)
            copy_label_with_flip(label_path, new_img_name, img.shape[1]) if os.path.exists(label_path) else None

        # 2. 亮度调整
        elif choice > 0.4:
            brightness_factor = random.uniform(0.8, 1.2)
            bright_img = adjust_brightness(img, brightness_factor)
            new_img_name = img_file.replace('.jpg', '_bright.jpg').replace('.png', '_bright.png').replace('.jpeg',
                                                                                                          '_bright.jpeg')
            new_img_path = os.path.join(images_dir, new_img_name)
            cv2.imwrite(new_img_path, bright_img)
            copy_label(label_path, new_img_name) if os.path.exists(label_path) else None

        # 3. 对比度调整
        elif choice > 0.3:
            contrast_factor = random.uniform(0.8, 1.2)
            contrast_img = adjust_contrast(img, contrast_factor)
            new_img_name = img_file.replace('.jpg', '_contrast.jpg').replace('.png', '_contrast.png').replace('.jpeg',
                                                                                                              '_contrast.jpeg')
            new_img_path = os.path.join(images_dir, new_img_name)
            cv2.imwrite(new_img_path, contrast_img)
            copy_label(label_path, new_img_name) if os.path.exists(label_path) else None

        # 4. 高斯噪声
        elif choice > 0.2:
            noise_img = add_gaussian_noise(img)
            new_img_name = img_file.replace('.jpg', '_noise.jpg').replace('.png', '_noise.png').replace('.jpeg',
                                                                                                        '_noise.jpeg')
            new_img_path = os.path.join(images_dir, new_img_name)
            cv2.imwrite(new_img_path, noise_img)
            copy_label(label_path, new_img_name) if os.path.exists(label_path) else None

        # 5. 旋转
        elif choice > 0.1:
            angle = random.uniform(-15, 15)
            rotated_img = rotate_image(img, angle)
            new_img_name = img_file.replace('.jpg', '_rot.jpg').replace('.png', '_rot.png').replace('.jpeg',
                                                                                                    '_rot.jpeg')
            new_img_path = os.path.join(images_dir, new_img_name)
            cv2.imwrite(new_img_path, rotated_img)
            copy_label(label_path, new_img_name) if os.path.exists(label_path) else None

        # 6. 对阴雨图像：双算法实现去雾+对比度增强
        else:
            dehazed_img = dehaze_image(img)
            enhanced_img = adjust_contrast(dehazed_img, random.uniform(1.1, 1.3))
            new_img_name = img_file.replace('.jpg', '_dehaze_contrast.jpg').replace('.png',
                                                                                    '_dehaze_contrast.png').replace(
                '.jpeg', '_dehaze_contrast.jpeg')
            new_img_path = os.path.join(images_dir, new_img_name)
            cv2.imwrite(new_img_path, enhanced_img)
            if not dehaze_printed:
                print(f"🌧️ 对阴雨图像：使用双算法实现去雾+对比度增强")
                dehaze_printed = True
            copy_label(label_path, new_img_name) if os.path.exists(label_path) else None

        # 7. 对逆光图像：自适应gamma校正
        if random.random() > 0.7:  # 30%概率应用gamma校正
            gamma_corrected_img = adaptive_gamma_correction(img)
            new_img_name = img_file.replace('.jpg', '_gamma.jpg').replace('.png', '_gamma.png').replace('.jpeg',
                                                                                                        '_gamma.jpeg')
            new_img_path = os.path.join(images_dir, new_img_name)
            cv2.imwrite(new_img_path, gamma_corrected_img)
            if not gamma_printed:
                print(f"🌅 对逆光图像：使用自适应gamma校正")
                gamma_printed = True
            copy_label(label_path, new_img_name) if os.path.exists(label_path) else None

        augmented_count += 1

    print(f"✅ 数据增强完成，新增 {augmented_count} 张增强图像")


def dehaze_image(img):
    """使用暗通道先验算法进行去雾"""
    img_float = img.astype(np.float64) / 255.0
    dark_channel = np.min(img_float, axis=2)
    img_size = img_float.shape[:2]
    num_brightest = int(0.001 * img_size[0] * img_size[1])
    dark_vec = dark_channel.reshape(-1)
    indices = np.argsort(dark_vec)[::-1][:num_brightest]
    brightest_pixels = img_float.reshape(-1, 3)[indices]
    A = np.max(brightest_pixels, axis=0)
    omega = 0.95
    t = 1 - omega * dark_channel
    t = np.maximum(t, 0.1)
    img_dehazed = np.zeros_like(img_float)
    for i in range(3):
        img_dehazed[:, :, i] = (img_float[:, :, i] - A[i]) / t + A[i]
    img_dehazed = np.clip(img_dehazed, 0, 1)
    return (img_dehazed * 255).astype(np.uint8)


def adaptive_gamma_correction(img):
    """自适应gamma校正"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    if mean_brightness < 100:
        gamma = 0.6 + random.uniform(0, 0.2)
    elif mean_brightness > 155:
        gamma = 1.2 + random.uniform(0, 0.2)
    else:
        gamma = 0.9 + random.uniform(0, 0.2)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)


def copy_label(original_label_path, new_img_name):
    """复制标签文件"""
    new_label_name = new_img_name.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt')
    new_label_path = original_label_path.replace(os.path.basename(original_label_path), new_label_name)
    with open(original_label_path, 'r') as src:
        content = src.read()
    with open(new_label_path, 'w') as dst:
        dst.write(content)


def copy_label_with_flip(original_label_path, new_img_name, img_width):
    """复制标签文件并修改水平翻转后的坐标"""
    new_label_name = new_img_name.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt')
    new_label_path = original_label_path.replace(os.path.basename(original_label_path), new_label_name)
    with open(original_label_path, 'r') as src:
        lines = src.readlines()
    with open(new_label_path, 'w') as dst:
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = parts[0]
                x_center = 1.0 - float(parts[1])  # 水平翻转：1 - x_center
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                width = max(0.0, min(1.0, width))
                height = max(0.0, min(1.0, height))
                dst.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            else:
                dst.write(line)


def adjust_brightness(img, factor):
    """调整图像亮度"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = hsv.astype(np.float64)
    hsv[:, :, 1] = hsv[:, :, 1] * factor
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def adjust_contrast(img, factor):
    """调整图像对比度"""
    img_contrast = img.astype(np.float64)
    img_contrast = (img_contrast - 127.5) * factor + 127.5
    img_contrast = np.clip(img_contrast, 0, 255)
    img_contrast = img_contrast.astype(np.uint8)
    return img_contrast


def add_gaussian_noise(img, mean=0, std=25):
    """添加高斯噪声"""
    noise = np.random.normal(mean, std, img.shape).astype(np.float32)
    noisy_img = img.astype(np.float32) + noise
    noisy_img = np.clip(noisy_img, 0, 255)
    return noisy_img.astype(np.uint8)


def rotate_image(img, angle):
    """旋转图像"""
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, rotation_matrix, (w, h))


def split_dataset(dataset_path):
    """数据集划分：训练集70%，验证集20%，测试集10%"""
    print("🔄 开始数据集划分...")

    images_dir = os.path.join(dataset_path, 'images')
    labels_dir = os.path.join(dataset_path, 'labels')

    # 获取所有图像文件
    all_image_files = []
    for root, dirs, files in os.walk(images_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                if not any(subdir in root for subdir in ['train', 'val', 'test']):
                    all_image_files.append(os.path.relpath(os.path.join(root, file), images_dir))

    # 过滤掉没有对应标签的图像
    valid_files = []
    for img_file in all_image_files:
        label_name = img_file.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt')
        label_path = os.path.join(labels_dir, label_name)
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                content = f.read().strip()
                if content:
                    valid_files.append(img_file)

    print(f"✅ 找到 {len(valid_files)} 个有标签的图像")

    if len(valid_files) == 0:
        raise ValueError("❌ 没有找到任何带标签的图像！请检查标签文件是否正确。")

    # 随机打乱
    random.shuffle(valid_files)

    # 创建输出目录
    for dir_path in ['images/train', 'images/val', 'images/test', 'labels/train', 'labels/val', 'labels/test']:
        os.makedirs(os.path.join(dataset_path, dir_path), exist_ok=True)

    # 计算划分比例
    total = len(valid_files)
    train_size = int(0.7 * total)
    val_size = int(0.2 * total)

    # 分割数据
    train_files = valid_files[:train_size]
    val_files = valid_files[train_size:train_size + val_size]
    test_files = valid_files[train_size + val_size:]

    # 移动文件
    move_files(train_files, images_dir, labels_dir, f'{dataset_path}/images/train', f'{dataset_path}/labels/train')
    move_files(val_files, images_dir, labels_dir, f'{dataset_path}/images/val', f'{dataset_path}/labels/val')
    move_files(test_files, images_dir, labels_dir, f'{dataset_path}/images/test', f'{dataset_path}/labels/test')

    print(f"✅ 数据集划分完成: 训练集{len(train_files)}张，验证集{len(val_files)}张，测试集{len(test_files)}张")


def move_files(file_list, src_img_dir, src_lbl_dir, dst_img_dir, dst_lbl_dir):
    """移动文件到目标目录"""
    for file_name in file_list:
        src_img_path = os.path.join(src_img_dir, file_name)
        dst_img_path = os.path.join(dst_img_dir, file_name)
        if os.path.exists(src_img_path):
            shutil.copy2(src_img_path, dst_img_path)
        label_name = file_name.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt')
        src_lbl_path = os.path.join(src_lbl_dir, label_name)
        dst_lbl_path = os.path.join(dst_lbl_dir, label_name)
        if os.path.exists(src_lbl_path):
            shutil.copy2(src_lbl_path, dst_lbl_path)


def test_model_on_test_set():
    """使用训练好的模型在测试集上进行评估"""
    print("🧪 开始在测试集上测试模型...")

    model_path = r'E:\AI_Training\City_Competition\code\runs\detect\train\weights\best.pt'
    dataset_path = r'E:\AI_Training\City_Competition\code\dataset\dataset1\data.yaml'

    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return

    try:
        model = YOLO(model_path)
        results = model.val(data=dataset_path, split='test', workers=0, plots=False)

        if hasattr(results, 'box') and results.box:
            map50 = getattr(results.box, 'map50', 0)
            map5095 = getattr(results.box, 'map', 0)
            precision = getattr(results.box, 'precision', 0)
            recall = getattr(results.box, 'recall', 0)

            print(f"📈 mAP@0.5 (测试集): {map50:.4f}")
            print(f"📈 mAP@0.5:0.95 (测试集): {map5095:.4f}")
            print(f"📈 Precision (测试集): {precision:.4f}")
            print(f"📈 Recall (测试集): {recall:.4f}")

            # 生成JSON结果
            test_results = {
                "model_path": model_path,
                "dataset_path": dataset_path,
                "metrics": {
                    "mAP50": float(map50),
                    "mAP50_95": float(map5095),
                    "precision": float(precision),
                    "recall": float(recall)
                }
            }

            # 保存JSON结果
            json_path = os.path.join(os.path.dirname(model_path), 'test_results.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(test_results, f, ensure_ascii=False, indent=2)
            print(f"📄 JSON测试结果已保存: {json_path}")

            return map50, map5095, precision, recall
        else:
            print("❌ 无法获取模型测试结果")
            return 0, 0, 0, 0

    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        return 0, 0, 0, 0


def main():
    """主函数 - 先增强数据，再划分数据集"""
    print("🚀 开始数据处理流程：数据增强 -> 数据集划分")

    # 数据集路径
    dataset_path = r"E:\AI_Training\City_Competition\code\dataset\dataset1"

    # 询问是否需要数据增强
    print("🔍 检查是否需要数据增强...")
    augment_choice = input("是否进行数据增强？(y/n，默认为y): ").lower().strip()
    if augment_choice in ['', 'y', 'yes']:
        augment_data(dataset_path, augment_count=100)
        print("✅ 数据增强完成")

    # 询问是否需要数据集划分
    print("\n🔍 检查是否需要数据集划分...")
    split_choice = input("是否进行数据集划分？(y/n，默认为y): ").lower().strip()
    if split_choice in ['', 'y', 'yes']:
        split_dataset(dataset_path)
        print("✅ 数据集划分完成")

    # 询问是否需要在测试集上测试模型
    print("\n🔍 检查是否需要在测试集上测试模型...")
    test_choice = input("是否在测试集上测试训练好的模型？(y/n，默认为y): ").lower().strip()
    if test_choice in ['', 'y', 'yes']:
        test_map50, test_map5095, test_precision, test_recall = test_model_on_test_set()
        print(f"\n🎯 测试集评估结果:")
        print(f"   mAP@0.5: {test_map50:.4f}")
        print(f"   mAP@0.5:0.95(0.5到0.95的平均mAP): {test_map5095:.4f}")
        print(f"   Precision(精确率): {test_precision:.4f}")
        print(f"   Recall(召回率): {test_recall:.4f}")
        print("✅ 模型测试完成")

    print("✅ 完整数据处理流程完成")


if __name__ == "__main__":
    main()