import os
import shutil
import yaml
from glob import glob
import random

def prepare_data():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)["prepare"]
    
    raw_dir = "data/raw"
    processed_train = "data/processed/train"
    processed_test = "data/processed/test"

    for d in [processed_train, processed_test]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    classes = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
    selected_classes = []
    for cow_class in classes:
        cow_path = os.path.join(raw_dir, cow_class)
        images = glob(os.path.join(cow_path, "*.jpg")) + glob(os.path.join(cow_path, "*.png")) + glob(os.path.join(cow_path, "*.jpeg"))

        if len(images) < params["min_images"]:
            print(f"Skipping {cow_class}: only {len(images)} images found.")
            continue
        selected_classes.append(cow_class)
        random.shuffle(images)
        split_idx = int(len(images) * (1 - params["split"]))
        train_images = images[:split_idx]
        test_images = images[split_idx:]
        
        # 5. Save to processed directories
        for img_list, target_root in [(train_images, processed_train), (test_images, processed_test)]:
            target_path = os.path.join(target_root, cow_class)
            os.makedirs(target_path, exist_ok=True)
            for img in img_list:
                shutil.copy(img, target_path)

    print("Data preparation complete: 90/10 split applied.")

if __name__ == "__main__":
    prepare_data()