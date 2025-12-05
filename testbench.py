from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolo11s.pt")

    model.info()

    results = model.train(
        data="fire.yaml",
        epochs=50,
        imgsz=640,
        batch=32,
    )

