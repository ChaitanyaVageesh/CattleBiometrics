import os
import shutil
import yaml
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# Assisted by Gemini

def offline_augmentation():
    with open("params.yaml") as f:
        config = yaml.safe_load(f)["transform"]
    
    in_train = "data/processed/train"
    out_train = "data/augmented/train"
    in_test = "data/processed/test"
    out_test = "data/augmented/test"
    
    # Clean up old directories to prevent ghost files
    for path in [out_train, out_test]:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

    # 1. Copy Test Data Exactly As-Is (Test data is NEVER augmented)
    print("Copying test data...")
    shutil.copytree(in_test, out_test, dirs_exist_ok=True)

    # 2. CHECK THE PARAMETER FLAG
    if not config.get('enable', True):
        print("Augmentation is DISABLED in params.yaml. Copying training data as-is...")
        shutil.copytree(in_train, out_train, dirs_exist_ok=True)
        print("Transform stage complete (Pass-through mode).")
        return  # Exit the script early!

    # 3. Augment Training Data Offline (Only runs if enable: True)
    print(f"Augmentation ENABLED. Generating {config['copies_per_image']} copies per image...")
    
    datagen = ImageDataGenerator(
        rotation_range=config['rotation_range'],
        width_shift_range=config['width_shift'],
        height_shift_range=config['height_shift'],
        shear_range=config['shear_range'],
        zoom_range=config['zoom_range'],
        brightness_range=config['brightness_range'],
        horizontal_flip=config['horizontal_flip'],
        fill_mode='nearest'
    )

    classes = os.listdir(in_train)
    
    for cls in classes:
        cls_in_path = os.path.join(in_train, cls)
        cls_out_path = os.path.join(out_train, cls)
        os.makedirs(cls_out_path, exist_ok=True)
        
        for img_name in os.listdir(cls_in_path):
            img_path = os.path.join(cls_in_path, img_name)
            
            # Save the original image first
            shutil.copy(img_path, os.path.join(cls_out_path, f"orig_{img_name}"))
            
            # Load image for Keras Datagen
            img = load_img(img_path)
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)
            
            # Generate and save copies
            i = 0
            for batch in datagen.flow(x, batch_size=1, save_to_dir=cls_out_path, 
                                      save_prefix=f"aug_{img_name.split('.')[0]}", save_format='jpeg'):
                i += 1
                if i >= config['copies_per_image']:
                    break

    print("Offline augmentation complete!")

if __name__ == "__main__":
    offline_augmentation()