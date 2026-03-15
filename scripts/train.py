

import os
import yaml
import pickle
import numpy as np
import cv2
from glob import glob
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_class_weight
from skimage.feature import hog

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet169
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Assisted by Gemini

class CattleTrainer:
    def __init__(self):
        with open("params.yaml") as f:
            self.config = yaml.safe_load(f)
        
        self.train_path = "data/augmented/train" 
        self.img_size = tuple(self.config['prepare']['img_size'])
        self.model_type = self.config['train']['model_type']
        self.classes = sorted(os.listdir(self.train_path))
        self.num_classes = len(self.classes)

    def get_datagen(self):
        """
        DenseNet169 expects specific preprocessing via preprocess_input.
        """
        return ImageDataGenerator(preprocessing_function=preprocess_input)

    def load_traditional_data(self):
        X, y = [], []
        print(f"Loading data for {self.model_type} from {self.train_path}...")
        
        for idx, cls_name in enumerate(self.classes):
            img_paths = glob(os.path.join(self.train_path, cls_name, "*.jp*g")) + \
                        glob(os.path.join(self.train_path, cls_name, "*.png"))
            
            for path in img_paths:
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None: continue
                
                img_resized = cv2.resize(img, self.img_size)
                
                if self.model_type == "svm":
                    features = hog(img_resized, orientations=9, pixels_per_cell=(16, 16),
                                   cells_per_block=(2, 2), visualize=False)
                    X.append(features)
                else:
                    X.append(img_resized.flatten() / 255.0)
                
                y.append(idx)
                
        return np.array(X), np.array(y)

    def get_class_weights(self, y_train):
        weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
        return dict(enumerate(weights))

    def train_svm(self):
        print("--- Training SVM ---")
        X_train, y_train = self.load_traditional_data()
        class_weights = self.get_class_weights(y_train)
        
        model = SVC(kernel='rbf', class_weight=class_weights, probability=True)
        model.fit(X_train, y_train)
        return model

    def train_mlp(self):
        print("--- Training MLP ---")
        X_train, y_train = self.load_traditional_data()
        
        model = MLPClassifier(
            hidden_layer_sizes=(self.config['train']['hidden_units'], 256),
            activation='relu', solver='adam', 
            max_iter=self.config['train']['epochs'], early_stopping=True
        )
        model.fit(X_train, y_train)
        return model

    def train_transfer(self):
        print("--- Training Transfer Learning Model (DenseNet169 Fine-Tuned) ---")
        
        train_gen = self.get_datagen().flow_from_directory(
            self.train_path, target_size=self.img_size,
            batch_size=self.config['train']['batch_size'], class_mode='categorical',
            shuffle=True
        )

        class_weights = self.get_class_weights(train_gen.classes)

        # Initialize DenseNet169
        base_model = DenseNet169(weights='imagenet', include_top=False, input_shape=(*self.img_size, 3))
        
        # --- SAFE FINE-TUNING ---
        # We unfreeze only the top 40 layers. This allows it to learn biometric ridges 
        # without destroying the 23 million pre-trained parameters below it.
        base_model.trainable = True
        for layer in base_model.layers[:-40]:
            layer.trainable = False

        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.5), 
            layers.Dense(self.num_classes, activation='softmax')
        ])

        # --- LABEL SMOOTHING --- 
        # Prevents overconfidence and massive overfitting on small datasets
        loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

        model.compile(optimizer=optimizers.Adam(learning_rate=1e-4),
                      loss=loss_fn, metrics=['accuracy'])
        
        callbacks = [
            ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2, min_lr=1e-6),
            EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        ]
        
        # We cap epochs at 20. DenseNet converges incredibly fast.
        model.fit(train_gen, epochs=min(self.config['train']['epochs'], 20), 
                  class_weight=class_weights, callbacks=callbacks)
        return model

    def run(self):
        os.makedirs("models", exist_ok=True)
        
        if self.model_type == "transfer":
            model = self.train_transfer()
            # Saving as .keras is the modern, crash-free standard
            model.save(f"models/model_{self.model_type}.keras")
            print(f"Saved to models/model_{self.model_type}.keras")
        else:
            if self.model_type == "svm":
                model = self.train_svm()
            elif self.model_type == "mlp":
                model = self.train_mlp()
            
            with open(f"models/model_{self.model_type}.pkl", "wb") as f:
                pickle.dump(model, f)
            print(f"Saved to models/model_{self.model_type}.pkl")

if __name__ == "__main__":
    trainer = CattleTrainer()
    trainer.run()