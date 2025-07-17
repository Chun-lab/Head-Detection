from ultralytics import YOLO

# 加载模型
model = YOLO("/home/liuchun/data/ultralytics-main/ultralytics/cfg/models/v8/yolov8.yaml").load("/home/liuchun/data/ultralytics-main/yolov8n.pt")  # 从YAML构建并转移权重

if __name__ == '__main__':
    # 训练模型
    results = model.train(data='/home/liuchun/data/ultralytics-main/ultralytics/cfg/datasets/deng.yaml', epochs=200, imgsz=640)


