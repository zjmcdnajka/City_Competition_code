import streamlit as st
import cv2
import numpy as np
import os
from ultralytics import YOLO
import tempfile
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from fpdf import FPDF
import io
from collections import defaultdict, deque

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class GarbageDetectionApp:
    def __init__(self):
        self.model = None
        self.tracking_history = defaultdict(lambda: deque(maxlen=5))  # 轨迹跟踪
        self.heatmap = None
        self.detection_results = []
        self.waste_names = {0: 'plasticBag', 1: 'plasticBottle', 2: 'polyfoam'}
        self.colors = {
            'plasticBag': (255, 0, 0),  # 红色
            'plasticBottle': (0, 255, 0),  # 绿色
            'polyfoam': (0, 0, 255)  # 蓝色
        }

    def load_model(self):
        """加载预训练模型"""
        if self.model is None:
            model_path = r'E:\AI_Training\City_Competition\code\runs\detect\train\weights\best.pt'
            if os.path.exists(model_path):
                self.model = YOLO(model_path)
                st.success("✅ 模型加载成功")
            else:
                st.error(f"❌ 模型文件不存在: {model_path}")
                return False
        return True

    def detect_frame(self, frame):
        """对单帧进行检测"""
        if self.model is None:
            return frame, []

        results = self.model.predict(frame, conf=0.5, verbose=False)
        detections = []

        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())

                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': conf,
                        'class_id': cls,
                        'class_name': self.waste_names[cls]
                    })

        return frame, detections

    def draw_detections(self, frame, detections):
        """在帧上绘制检测结果"""
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            conf = detection['confidence']
            class_name = detection['class_name']
            color = self.colors[class_name]

            # 绘制边界框
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # 绘制标签
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame

    def update_tracking(self, detections, frame_num):
        """更新轨迹跟踪"""
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            class_name = detection['class_name']

            # 简单的ID分配（基于位置）
            obj_id = f"{class_name}_{center_x}_{center_y}_{frame_num}"
            self.tracking_history[obj_id].append((center_x, center_y, frame_num))

    def draw_tracking(self, frame, selected_obj_id=None):
        """绘制轨迹"""
        for obj_id, positions in self.tracking_history.items():
            if len(positions) > 1:
                for i in range(1, len(positions)):
                    start_pos = positions[i - 1][:2]
                    end_pos = positions[i][:2]

                    # 如果是选中的目标，用不同颜色绘制轨迹
                    if selected_obj_id and obj_id == selected_obj_id:
                        color = (255, 255, 255)  # 白色轨迹
                        thickness = 3
                    else:
                        color = (255, 255, 0)  # 黄色轨迹
                        thickness = 1

                    cv2.line(frame, start_pos, end_pos, color, thickness)

        return frame

    def update_heatmap(self, frame_shape, detections):
        """更新热力图"""
        if self.heatmap is None:
            self.heatmap = np.zeros((frame_shape[0], frame_shape[1]), dtype=np.float32)

        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # 在中心点周围增加热力值
            for dx in range(-20, 21):
                for dy in range(-20, 21):
                    nx, ny = center_x + dx, center_y + dy
                    if 0 <= nx < self.heatmap.shape[1] and 0 <= ny < self.heatmap.shape[0]:
                        distance = np.sqrt(dx ** 2 + dy ** 2)
                        if distance <= 20:
                            self.heatmap[ny, nx] += (20 - distance) / 20

        return self.heatmap

    def generate_report_data(self, detections, frame_shape):
        """生成报告数据"""
        report_data = []
        total_count = len(detections)
        class_counts = defaultdict(int)

        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            class_name = detection['class_name']
            conf = detection['confidence']

            # 计算面积（作为堆积量的参考）
            area = (x2 - x1) * (y2 - y1)

            report_data.append({
                'Type': class_name,
                'X': (x1 + x2) // 2,
                'Y': (y1 + y2) // 2,
                'Confidence': conf,
                'Area': area,
                'BoundingBox': f"[{x1},{y1},{x2},{y2}]"
            })

            class_counts[class_name] += 1

        return report_data, total_count, dict(class_counts)

    def create_pdf_report(self, report_data, total_count, class_counts):
        """创建PDF报告"""
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, '垃圾分类检测报告', 0, 1, 'C')

        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 10, f'总检测数量: {total_count}', 0, 1)

        pdf.cell(0, 10, '各类垃圾数量统计:', 0, 1)
        for class_name, count in class_counts.items():
            pdf.cell(0, 8, f'  {class_name}: {count}', 0, 1)

        pdf.cell(0, 10, '检测详情:', 0, 1)
        pdf.set_font('Arial', '', 10)

        for i, detection in enumerate(report_data[:50]):  # 限制显示前50个
            pdf.cell(0, 6,
                     f"  {i + 1}. 类型: {detection['Type']}, 坐标: ({detection['X']}, {detection['Y']}), 置信度: {detection['Confidence']:.2f}, 面积: {detection['Area']}",
                     0, 1)

        return pdf.output(dest='S').encode('latin-1')


def main():
    st.set_page_config(page_title="垃圾分类检测系统", layout="wide")
    st.title("🗑️ 垃圾分类YOLOv8检测系统")

    app = GarbageDetectionApp()

    # 侧边栏
    st.sidebar.header("⚙️ 系统设置")

    # 模型加载
    if st.sidebar.button("加载模型"):
        if app.load_model():
            st.sidebar.success("模型加载成功！")
        else:
            st.sidebar.error("模型加载失败！")

    # 上传视频
    uploaded_file = st.sidebar.file_uploader("📁 上传视频文件", type=['mp4', 'avi', 'mov'])

    if uploaded_file is not None:
        # 保存上传的视频到临时文件
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        # 读取视频
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        st.sidebar.success(f"视频加载成功！帧数: {frame_count}, FPS: {fps:.2f}")

        # 选择功能
        st.sidebar.header("🔧 选择功能")
        show_detections = st.sidebar.checkbox("显示边界框", value=True)
        show_tracking = st.sidebar.checkbox("显示轨迹跟踪", value=True)
        show_heatmap = st.sidebar.checkbox("显示热力图", value=True)
        generate_report = st.sidebar.checkbox("生成报告", value=True)

        # 处理视频
        if st.sidebar.button("开始处理"):
            st.header("🎬 视频处理中...")
            progress_bar = st.progress(0)

            frame_num = 0
            all_detections = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 检测
                processed_frame, detections = app.detect_frame(frame)

                # 绘制检测结果
                if show_detections:
                    processed_frame = app.draw_detections(processed_frame, detections)

                # 更新轨迹
                if show_tracking:
                    app.update_tracking(detections, frame_num)
                    processed_frame = app.draw_tracking(processed_frame)

                # 更新热力图
                if show_heatmap:
                    heatmap = app.update_heatmap(frame.shape, detections)

                all_detections.extend(detections)
                frame_num += 1

                # 更新进度
                progress = frame_num / frame_count
                progress_bar.progress(progress)

                # 显示当前帧（可选，为了性能考虑可以注释掉）
                # stframe.image(processed_frame, channels="BGR", use_column_width=True)

            cap.release()
            st.success("✅ 视频处理完成！")

            # 显示结果
            if generate_report:
                st.header("📊 检测结果统计")

                # 生成报告数据
                report_data, total_count, class_counts = app.generate_report_data(all_detections, frame.shape)

                # 显示统计信息
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("总检测数量", total_count)
                with col2:
                    st.metric("塑料袋数量", class_counts.get('plasticBag', 0))
                with col3:
                    st.metric("塑料瓶数量", class_counts.get('plasticBottle', 0))

                col4, col5 = st.columns(2)
                with col4:
                    st.metric("泡沫数量", class_counts.get('polyfoam', 0))
                with col5:
                    st.metric("平均置信度",
                              f"{np.mean([d['confidence'] for d in all_detections]):.2f}" if all_detections else 0)

                # 显示检测详情
                if report_data:
                    st.subheader("📋 检测详情")
                    df = pd.DataFrame(report_data)
                    st.dataframe(df.head(20))  # 显示前20个

                # 生成PDF报告
                if st.button("📄 生成PDF报告"):
                    pdf_bytes = app.create_pdf_report(report_data, total_count, class_counts)
                    st.download_button(
                        label="📥 下载PDF报告",
                        data=pdf_bytes,
                        file_name="垃圾分类检测报告.pdf",
                        mime="application/pdf"
                    )

            # 显示热力图
            if show_heatmap and app.heatmap is not None:
                st.header("🌡️ 垃圾密度热力图")
                fig, ax = plt.subplots(figsize=(10, 8))

                # 创建自定义颜色映射（红色表示高风险）
                colors = ['blue', 'yellow', 'red']
                n_bins = 256
                cmap = LinearSegmentedColormap.from_list('risk_heatmap', colors, N=n_bins)

                im = ax.imshow(app.heatmap, cmap=cmap, alpha=0.7)
                plt.colorbar(im, ax=ax, label='垃圾密度')
                ax.set_title('垃圾密度热力图（红色为高风险区域）')

                st.pyplot(fig)

        # 清理临时文件
        os.unlink(tfile.name)


if __name__ == "__main__":
    main()