import os
import shutil
import random
from glob import glob

# Paths
image_root = "multiclass_ground_truth_images"  # c0, c1, ...
yolo_label_root = "yolo_labels"
output_root = "data_train-test"
train_ratio = 0.8

# Ensure output subfolders exist
for split in ["train", "test"]:
    os.makedirs(os.path.join(output_root, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_root, split, "labels"), exist_ok=True)

# Collect all label files
all_labels = glob(os.path.join(yolo_label_root, "*.txt"))
frames = sorted(list({os.path.basename(f).split("_")[0] for f in all_labels}))
random.shuffle(frames)

# Split frames into train/test
num_train = int(len(frames) * train_ratio)
train_frames = set(frames[:num_train])
test_frames = set(frames[num_train:])

print(f"Total frames: {len(frames)}, Train: {len(train_frames)}, Test: {len(test_frames)}")

# Copy labels and corresponding images
for label_file in all_labels:
    base = os.path.basename(label_file)
    frame_num, cam_part = base.split("_")
    cam_num = cam_part.split(".")[0]
    
    split = "train" if frame_num in train_frames else "test"
    
    img_src = os.path.join(image_root, cam_num, f"{frame_num}.jpg")
    img_dst = os.path.join(output_root, split, "images", f"{frame_num}_{cam_num}.jpg")
    label_dst = os.path.join(output_root, split, "labels", f"{frame_num}_{cam_num}.txt")
    
    # Skip if image is missing
    if not os.path.exists(img_src):
        print(f"⚠️ Image missing: {img_src}")
        continue
    
    shutil.copy2(img_src, img_dst)
    shutil.copy2(label_file, label_dst)

print("Train/test split completed successfully!")