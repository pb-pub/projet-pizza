import cv2
import hough
import os
import numpy as np
from sklearn.model_selection import train_test_split, KFold
import knn
import gmm
from skimage.io import imread
import matplotlib.pyplot as plt

import descripteurs_texture as dt
import descripteur_couleur as ct

def pre_process_image(image_path):
    masked_image = hough.mask_pizza(image_path)    
    return masked_image

def process_features(masked_image):
    features = []
    texture_features = dt.texture_features(masked_image)
    color_features = ct.extract_color_features(masked_image)
    
    features = np.concatenate((texture_features, color_features))
    return features 

def kfold_evaluation_knn(X, y, k_values, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_precisions = []
    fold_recalls = []
    fold_f1s = []
    
    for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
        print(f"Processing fold {fold}/{n_splits}...")
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        precisions, recalls, f1s = knn.evaluate_knn(X_train, X_test, y_train, y_test, k_values)
        fold_precisions.append(precisions)
        fold_recalls.append(recalls)
        fold_f1s.append(f1s)
    
    avg_precisions = np.mean(fold_precisions, axis=0)
    avg_recalls = np.mean(fold_recalls, axis=0)
    avg_f1s = np.mean(fold_f1s, axis=0)
    
    std_precisions = np.std(fold_precisions, axis=0)
    std_recalls = np.std(fold_recalls, axis=0)
    std_f1s = np.std(fold_f1s, axis=0)
    
    return avg_precisions, avg_recalls, avg_f1s, std_precisions, std_recalls, std_f1s

if __name__ == "__main__":
    choice = 2
    if choice == 1:
        print("KNN")
    elif choice == 2:
        print("SVM")
    else:
        print("Invalid choice")
        exit()
    
    folders = ['pizzafromag', 'pizzahawai', 'pizzamargherita', 'pizzapepperoni', 'pizzareine', 'pizzavege']
    base_dir = "masked"
    
    features = []
    labels = []
    
    for i in range(len(folders)):
        dataset_path = os.path.join(base_dir, folders[i])
        
        files = [f for f in os.listdir(dataset_path) if f.endswith('.jpg')]
        for file in files:
            
            img_path = os.path.join(dataset_path, file)
            
            img = imread(img_path)
            feat_vec = process_features(img)
            features.append(feat_vec)
            labels.append(i + 1)
     
    X = np.array(features)
    y = np.array(labels)
    
    np.save('features.npy', features)
    np.save('labels.npy', labels)
    
    n_folds = 5
    use_kfold = True
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if choice == 1:
        k_values = np.arange(8, 15)
        
        if use_kfold:
            print(f"Performing {n_folds}-fold cross-validation for KNN...")
            avg_precisions, avg_recalls, avg_f1s, std_precisions, std_recalls, std_f1s = kfold_evaluation_knn(X, y, k_values, n_folds)
            
            plt.figure(figsize=(12, 8))
            plt.errorbar(k_values, avg_f1s, yerr=std_f1s, label='F1-score', capsize=3)
            plt.errorbar(k_values, avg_precisions, yerr=std_precisions, label='Précision', capsize=3)
            plt.xlabel('k')
            plt.ylabel('Score')
            plt.title(f'Scores moyens avec {n_folds}-fold cross-validation')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.show()
            
            best_k_index = np.argmax(avg_f1s)
            best_k = k_values[best_k_index]
            print(f"\nMeilleur k selon la cross-validation {n_folds}-fold: {best_k}")
            print(f"F1-score moyen: {avg_f1s[best_k_index]:.4f} ± {std_f1s[best_k_index]:.4f}")
            print(f"Précision moyenne: {avg_precisions[best_k_index]:.4f} ± {std_precisions[best_k_index]:.4f}")
        else:
            precisions, recalls, f1s = knn.evaluate_knn(X_train, X_test, y_train, y_test, k_values)
            
            plt.plot(k_values, f1s, label='F1-score')
            plt.plot(k_values, precisions, label='Précision')
            plt.xlabel('k')
            plt.ylabel('Score')
            plt.title('Scores pour différentes valeurs de k (train-test split simple)')
            plt.legend()
            plt.show()

    elif choice == 2:
        results = gmm.evaluate_model(X_train, X_test, y_train, y_test, folders)
        print(results)
        gmm.visualize_confusion_matrix(results['confusion_matrix'], folders)