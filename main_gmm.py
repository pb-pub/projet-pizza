import cv2
import hough
import os
import numpy as np
from sklearn.model_selection import train_test_split
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
    texture_features = dt.texture_features(masked_image)
    color_features = ct.extract_color_features(masked_image)
    
    # Normaliser les caractéristiques individuellement
    texture_features = (texture_features - np.mean(texture_features)) / np.std(texture_features)
    color_features = (color_features - np.mean(color_features)) / np.std(color_features)
    
    # Combiner les caractéristiques normalisées
    features = np.concatenate((texture_features, color_features))
    return features



if __name__ == "__main__":
    # Load images
    # images = load_images("dataset")
    
    # test
    # image = pre_process_image("dataset/pizzamargherita/m15.jpg")
    # features = process_features(image)
    # print(features)

    print("GMM")
    
    folders = ['pizzafromag', 'pizzahawai', 'pizzamargherita', 'pizzapepperoni', 'pizzareine', 'pizzavege']
    base_dir = "masked_dataset_carre" #using this because the pre-processed images are in the masked folder
    # base_dir = "dataset" # uncomment this if you want to use the original images
    
    features = []
    labels = []
    
    
    for i in range(len(folders)):
        dataset_path = os.path.join(base_dir, folders[i])
        
        files = [f for f in os.listdir(dataset_path) if f.endswith('.jpg')]
        for file in files:
            
            img_path = os.path.join(dataset_path, file)
            
            #img = pre_process_image(img_path) # uncomment this if you want to use the original images
            img = imread(img_path)  # comment this if you want to use the original images
            feat_vec = process_features(img)
            features.append(feat_vec)
            labels.append(i)  # Changed from i + 1 to i
     
            
    X = np.array(features)
    y = np.array(labels)
    
    # save the features in a file 
    np.save('features.npy', features)
    np.save('labels.npy', labels)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if choice == 1:
        k_values = np.arange(1, 26)
        precisions, recalls, f1s = knn.evaluate_knn(X_train, X_test, y_train, y_test, k_values)
        
        #show the metrics for k on a plot
        # print(precisions)
        
        # plt.plot(k_values, recalls, label='Rappel')
        plt.plot(k_values, f1s, label='F1-score')
        plt.plot(k_values, precisions, label='Précision')
        plt.xlabel('k')
        plt.ylabel('Score')
        plt.title('Scores pour différentes valeurs de k (uniquement couleur et texture)')
        plt.legend()
        plt.show()

    elif choice == 2:
        results = gmm.evaluate_model(X_train, X_test, y_train, y_test, folders)
        print(results)
        gmm.visualize_confusion_matrix(results['confusion_matrix'], folders)