


import os
import yaml
import json
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, confusion_matrix
from skimage.feature import hog
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.densenet import preprocess_input

# Assisted by Gemini

def evaluate():
    with open("params.yaml") as f:
        config = yaml.safe_load(f)
    
    test_path = "data/augmented/test" 
    model_type = config['train']['model_type']
    img_size = tuple(config['prepare']['img_size'])
    classes = sorted(os.listdir(test_path))

    # 1. LOAD MODEL
    if model_type == "transfer":
        # Load the modern .keras format
        model_path = f"models/model_{model_type}.keras"
        print(f"Loading Keras model: {model_path}")
        model = load_model(model_path)
    else:
        model_path = f"models/model_{model_type}.pkl"
        print(f"Loading Scikit-Learn model: {model_path}")
        with open(model_path, "rb") as f:
            model = pickle.load(f)

    # 2. LOAD TEST DATA
    X_test, y_test = [], []
    for idx, cls in enumerate(classes):
        cls_path = os.path.join(test_path, cls)
        if not os.path.isdir(cls_path): continue
        
        for img_file in os.listdir(cls_path):
            f_path = os.path.join(cls_path, img_file)
            
            img = cv2.imread(f_path)
            if img is None: 
                continue
            
            if model_type == "svm":
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_resized = cv2.resize(img_gray, img_size)
                features = hog(img_resized, orientations=9, pixels_per_cell=(16, 16),
                               cells_per_block=(2, 2), visualize=False)
                X_test.append(features)
                
            elif model_type == "mlp":
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_resized = cv2.resize(img_gray, img_size)
                X_test.append(img_resized.flatten() / 255.0)
                
            else: # transfer (DenseNet169)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(img_rgb, img_size)
                # Apply DenseNet's specific math preprocessing
                img_preprocessed = preprocess_input(np.array(img_resized, dtype=np.float32))
                X_test.append(img_preprocessed)
            
            y_test.append(idx)

    X_test, y_test = np.array(X_test), np.array(y_test)

    # 3. PREDICT
    if model_type == "transfer":
        y_pred = np.argmax(model.predict(X_test), axis=1)
    else:
        y_pred = model.predict(X_test)

    # 4. METRICS & PLOTS
    score = f1_score(y_test, y_pred, average='macro')
    print(f"\n==========================================")
    print(f"🎯 Evaluation Complete! Macro F1-Score: {score:.4f}")
    print(f"==========================================\n")
    
    with open("metrics.json", "w") as f:
        json.dump({"macro_f1": float(score), "model": model_type}, f, indent=4)

    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', xticklabels=classes, yticklabels=classes)
    plt.title(f"Confusion Matrix: {model_type}")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig("plots/confusion_matrix.png")

if __name__ == "__main__":
    evaluate()