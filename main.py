import cv2
import hough
import os
import numpy as np
from sklearn.model_selection import train_test_split
import knn

import descripteurs_texture as dt
import descripteur_couleur as ct

def pre_process_image(image_path):
    masked_image = hough.mask_pizza(image_path)    
    return masked_image

def process_features(masked_image):
    features = []
    texture_features = dt.texture_features(masked_image)
    color_features = ct.extract_color_features(masked_image)
    
    # color_features = color_features.flatten()
    features = np.concatenate((texture_features, color_features))
    return color_features 

 

if __name__ == "__main__":
    # Load images
    # images = load_images("dataset")
    
    # test
    # image = pre_process_image("dataset/pizzamargherita/m15.jpg")
    # features = process_features(image)
    # print(features)
    
    folders = ['pizzafromag', 'pizzahawai', 'pizzamargherita', 'pizzapepperoni', 'pizzareine', 'pizzavege']
    base_dir = "masked"
    
    features = []
    labels = []
    
    for i, folder in enumerate(folders, 1):
        folder_path = os.path.join(base_dir, folder)
        print(f"Traitement du dossier: {folder_path}")
        
        files = os.listdir(folder_path)
        print(f"Nombre d'images trouvées dans {folder}: {len(files)}")
        
        for file in files:
            image_path = os.path.join(folder_path, file)
            # masked_image = pre_process_image(image_path)
            masked_image = cv2.imread(image_path)
            features.append(process_features(masked_image))
            labels.append(i + 1)
            
    
            
    X = np.array(features)
    y = np.array(labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    k_values = [2,4,6,8,10,12,14]
    knn.evaluate_knn(X_train, X_test, y_train, y_test, k_values)
    
    
