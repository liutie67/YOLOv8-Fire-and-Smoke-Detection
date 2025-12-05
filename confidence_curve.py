import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO


def plot_video_confidence(video_path, model_path, target_class_id=0):
    """
    video_path: 视频文件路径
    model_path: YOLO权重文件路径
    target_class_id: 目标类别的ID (例如 COCO数据集中: 0=Person, 2=Car, 5=Bus 等)
    """

    # 1. 加载模型
    model = YOLO(model_path)

    # 2. 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: 无法打开视频文件")
        return

    frames_indices = []  # 横坐标：帧数
    confidences = []  # 纵坐标：置信度
    frame_count = 0

    print(f"开始处理视频，目标类别ID: {target_class_id}...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 3. YOLO 推理 (verbose=False 防止控制台刷屏)
        results = model(frame, verbose=False)

        # 获取当前帧中所有的检测框
        # result.boxes.cls 是类别索引，result.boxes.conf 是置信度
        boxes = results[0].boxes

        max_conf = 0.0  # 默认该帧没有目标，概率为0

        if boxes is not None:
            # 筛选出属于 target_class_id 的所有框
            target_indices = (boxes.cls == target_class_id).nonzero(as_tuple=True)[0]

            if len(target_indices) > 0:
                # 找到这些目标中置信度最高的一个
                target_confs = boxes.conf[target_indices]
                max_conf = float(target_confs.max())

        # 记录数据
        frames_indices.append(frame_count)
        confidences.append(max_conf)

        # 可选：每100帧打印一次进度
        if frame_count % 100 == 0:
            print(f"已处理 {frame_count} 帧...")

    cap.release()
    print("处理完成，正在绘图...")

    # 4. 绘制曲线图
    plt.figure(figsize=(12, 6))
    plt.plot(frames_indices, confidences, label=f'Class {target_class_id} Confidence', color='blue', linewidth=1)

    plt.title(f'Prediction Confidence per Frame (Class ID: {target_class_id})')
    plt.xlabel('Frame Index')
    plt.ylabel('Confidence Score (Probability)')
    plt.ylim(-0.05, 1.05)  # 设置Y轴范围在0-1之间略微留白
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.show()


# --- 使用示例 ---
# 这里的 'video.mp4' 请换成你的视频路径
# target_class_id=0 代表 'person' (COCO数据集默认)
plot_video_confidence(
    video_path=r"path/to/folder",
    model_path="path/to/model",
    target_class_id=1,
)