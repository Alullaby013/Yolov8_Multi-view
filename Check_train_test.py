import os

train_labels = os.listdir("data_train-test/train/labels")
test_labels = os.listdir("data_train-test/test/labels")

train_frames = set([f.split("*")[0] for f in train_labels])
test_frames = set([f.split("*")[0] for f in test_labels])

# Check for overlap

overlap = train_frames.intersection(test_frames)
if overlap:
    print(f"⚠️ Frames in both train and test: {sorted(overlap)}")
else:
    print("✅ No frames are shared between train and test!")

# Count images and labels

train_images = os.listdir("data_train-test/train/images")
test_images = os.listdir("data_train-test/test/images")

print(f"Train images: {len(train_images)}, Train labels: {len(train_labels)}")
print(f"Test images: {len(test_images)}, Test labels: {len(test_labels)}")
