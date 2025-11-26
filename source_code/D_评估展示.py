from ultralytics import YOLO


def train_yolo_model():
    """训练YOLOv8模型"""
    print("🏗️ 开始构建和训练模型")

    # 加载预训练模型
    model = YOLO('E:\AI_Training\City_Competition\code\yolo11n.pt')

    # 训练参数
    train_args = {
        'data': r'E:\AI_Training\City_Competition\code\dataset\dataset1\data.yaml',
        # 数据集配置文件路径，包含训练集、验证集路径和类别信息
        'epochs': 100,  # 训练轮数，模型将遍历整个数据集100次
        'imgsz': 832,  # 输入图像尺寸，所有图像会被调整为832x832像素进行训练
        'rect': True,  # 启用矩形训练，根据批次中图像的实际宽高比进行调整，可提高效率
        'batch': 25,  # 批次大小，每次训练处理25张图像
        'device': '0',  # 指定GPU设备，'0'表示使用第一块GPU
        'save_period': 1,  # 每1个epoch保存一次模型权重和训练结果
        'project': '../runs/detect',  # 训练结果保存的项目目录
        'exist_ok': True,  # 允许覆盖已存在的训练目录，避免重复运行时的冲突
        'amp': True,  # 启用自动混合精度训练，减少显存占用并可能加快训练速度
        'workers': 0,  # 数据加载进程数，0表示使用主进程加载数据
        'lr0': 0.01,  # 初始学习率，控制模型参数更新的步长
        'lrf': 0.01,  # 最终学习率，训练结束时的学习率（学习率会从lr0衰减到lrf）
        'warmup_epochs': 3,  # 预热轮数，前3个epoch会逐渐增加学习率到初始值
        'cache': 'ram',  # 将数据集缓存到内存中，加快数据读取速度
        'close_mosaic': 10,  # 从第10个epoch开始关闭Mosaic数据增强（Mosaic增强在训练初期有效，后期可能影响精度）
        # 损失函数各部分的权重系数
        'box': 7.5,  # 边界框回归损失的权重
        'cls': 0.5,  # 分类损失的权重
        'dfl': 1.5  # 分布焦点损失的权重（用于边界框精确定位）
    }

    # 开始训练
    print("📊 正在训练模型...")
    try:
        results = model.train(**train_args)
        print("✅ 模型训练完成")
        return model
    except Exception as e:
        print(f"❌ 训练出错: {e}")
        return None


def evaluate_model():
    """评估模型性能（验证集）"""
    print("📊 开始评估模型（验证集）")

    # 模型路径
    model_path = r'E:\AI_Training\City_Competition\code\runs\detect\train\weights\best.pt'
    dataset_path = r'E:\AI_Training\City_Competition\code\dataset\dataset1\data.yaml'

    try:
        # 加载模型
        print(f"📦 加载模型: {model_path}")
        model = YOLO(model_path)

        # 验证模型（在验证集上评估）
        results = model.val(
            data=dataset_path,
            split='val',  # 指定在验证集上评估
            workers=0,
            plots=False  # 避免自动生成图像
        )

        # 提取关键指标 - 修正属性访问
        if hasattr(results, 'box') and results.box:
            map50 = getattr(results.box, 'map50', 0)
            map5095 = getattr(results.box, 'map', 0)

            print(f"📈 mAP@0.5 (验证集): {map50:.4f}")
            print(f"📈 mAP@0.5:0.95 (验证集): {map5095:.4f}")

            # 尝试获取其他指标（根据实际返回对象的属性）
            precision = getattr(results.box, 'precision', 0)
            recall = getattr(results.box, 'recall', 0)

            if precision and recall:
                print(f"📊 详细指标 (验证集):")
                print(f"   - Precision: {precision:.4f}")
                print(f"   - Recall: {recall:.4f}")

            return map50, map5095
        else:
            print("❌ 无法获取模型评估结果")
            return 0, 0

    except Exception as e:
        print(f"❌ 模型评估失败: {e}")
        return 0, 0


def main():
    """主函数 - 先训练模型，再进行评估和测试"""
    print("🚀 开始完整流程：训练 -> 评估 -> 测试")

    # 1. 训练模型
    trained_model = train_yolo_model()

    if trained_model is None:
        print("❌ 模型训练失败，无法进行评估和测试")
        return

    print("\n" + "=" * 50)

    # 2. 验证模型（在验证集上）
    print("🔍 开始模型验证（验证集）")
    val_map50, val_map5095 = evaluate_model()

    print(f"\n🎯 验证集评估结果:")
    print(f"   mAP@0.5: {val_map50:.4f}")
    print(f"   mAP@0.5:0.95: {val_map5095:.4f}")

    print("\n🎯 完整流程完成")
    print(f"📊 最终结果汇总:")
    print(f"   验证集 mAP@0.5: {val_map50:.4f}, mAP@0.5:0.95: {val_map5095:.4f}")


if __name__ == "__main__":
    main()