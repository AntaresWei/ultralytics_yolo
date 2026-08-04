from ultralytics import YOLO
import numpy as np

model = YOLO("weights\yolo26s-depth.pt")

results = model(r"data\20260716112715_Color.png")

for i, result in enumerate(results):
    depth_map = result.depth.data.cpu().numpy()

    np.save(f"depth_{i}.npy", depth_map)

print("Saved depth map.")