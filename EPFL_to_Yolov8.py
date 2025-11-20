import os
from glob import glob
import cv2

# Paths
image_root = "multiclass_ground_truth_images"
label_root = "multiclass_ground_truth/bounding_boxes_EPFL_cross"
yolo_label_root = "yolo_labels"

os.makedirs(yolo_label_root, exist_ok=True)

# Map classes to IDs
folder_to_class = {
    "gt_files242_person": 0,
    "gt_files242_car": 1,
    "gt_files242_bus": 2
}

def convert_to_yolo(x, y, w, h, img_w, img_h):
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w_norm = w / img_w
    h_norm = h / img_h
    return x_center, y_center, w_norm, h_norm

for folder_name, class_id in folder_to_class.items():
    label_path = os.path.join(label_root, folder_name, "visible_frame")
    label_files = glob(os.path.join(label_path, "*.txt"))

    for lf in label_files:
        base = os.path.basename(lf)
        parts = base.split("_")
        if len(parts) != 3:
            continue
        frame_num = int(parts[1].replace("frame", ""))
        cam_num = parts[2].replace(".txt","").replace("cam","")

        img_name = f"{frame_num:08d}.jpg"
        img_path = os.path.join(image_root, f"c{cam_num}", img_name)
        if not os.path.exists(img_path):
            print(f"⚠️ Image not found: {img_path}")
            continue

        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        yolo_lines = []
        with open(lf, "r") as f:
            for line in f:  
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                x_abs, y_abs, w_abs, h_abs = map(float, parts)
                x_c, y_c, w_n, h_n = convert_to_yolo(x_abs, y_abs, w_abs, h_abs, w, h)
                yolo_lines.append(f"{class_id} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}")

        yolo_file_path = os.path.join(yolo_label_root, f"{frame_num:08d}_c{cam_num}.txt")
        with open(yolo_file_path, "w") as f:
            f.write("\n".join(yolo_lines))

print("YOLO label conversion complete!")
