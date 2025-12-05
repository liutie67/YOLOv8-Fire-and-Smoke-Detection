import cv2
from ultralytics import YOLO
from tqdm import tqdm  # 引入进度条库
import os



if __name__ == "__main__":
    folder = 'yyy'
    filenames = ['xxx']

    for filename in filenames:
        model = YOLO("runs/detect/train3-11n/weights/best.pt")
        source = fr"C:\Users\imnew\projects\databases\mileage\{folder}\{filename}.MOV"

        save_dir = fr"C:\Users\imnew\projects\databases\mileage\{folder}(predicted)"

        # 1. 预先获取视频总帧数（为了给进度条计算百分比）
        cap = cv2.VideoCapture(source)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        print(f"视频总帧数: {total_frames}")

        # 2. 运行推理
        # verbose=False: 关键参数！关闭每帧的详细打印
        results = model(source,
                        stream=True,
                        save=True,
                        project=save_dir,
                        name="run_clean",
                        verbose=False)

        print("开始处理...")

        # 3. 使用 tqdm 包装 results 生成器
        # total=total_frames 让进度条知道什么时候结束
        # unit="frame" 显示单位
        for _ in tqdm(results, total=total_frames, unit="frame"):
            pass  # 这里的 pass 是必须的，它驱动了推理过程

        print("\n处理完成！")