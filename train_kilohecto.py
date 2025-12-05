from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolo11s.pt")

    model.info()

    results = model.train(
        data="kilohecto.yaml",
        epochs=100,
        imgsz=1920,
        batch=4,
        pretrained=True,
        patience=20,
    )

