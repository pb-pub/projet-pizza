import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import glob
from skimage.feature import graycomatrix, graycoprops
import knn

def calculate_texture_features(img_gray, distances, angles):
    """
    Calcule les caractéristiques de texture (GLCM) pour une image en niveaux de gris.
    
    Parameters:
    -----------
    img_gray : array-like
        Image en niveaux de gris
    distances : list
        Liste des distances pour la matrice de co-occurrence
    angles : list
        Liste des angles pour la matrice de co-occurrence
    
    Returns:
    --------
    list
        Liste des caractéristiques de texture (corrélation et énergie)
    """
    features = []
    for d in distances:
        glcm = graycomatrix(img_gray, distances=[d], angles=angles, levels=256, symmetric=True, normed=True)
        correlation = graycoprops(glcm, 'correlation').mean()
        energy = graycoprops(glcm, 'energy').mean()
        features.extend([correlation, energy])
    return features

def load_and_preprocess_images(base_dir, folders, distances):
    """
    Charge les images, les prétraite et calcule les caractéristiques de texture.
    
    Parameters:
    -----------
    base_dir : str
        Répertoire de base contenant les dossiers d'images
    folders : list
        Liste des dossiers contenant les images par catégorie
    distances : list
        Liste des distances pour la GLCM
    
    Returns:
    --------
    tuple
        Caractéristiques et étiquettes extraites (features, labels)
    """
    features = []
    labels = []
    
    for i, folder in enumerate(folders, 1):
        folder_path = os.path.join(base_dir, folder)
        print(f"Traitement du dossier: {folder_path}")
        
        files = glob.glob(os.path.join(folder_path, '*.jpg'))
        print(f"Nombre d'images trouvées dans {folder}: {len(files)}")
        
        for file in files:
            img = cv2.imread(file)
            if img is None:
                print(f"Impossible de charger l'image: {file}")
                continue
                
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_resized = cv2.resize(img_gray, (400, 400))
            mask_resized = img_resized > 0
            
            angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
            feat_vec = calculate_texture_features(img_resized, distances, angles)
            
            features.append(feat_vec)
            labels.append(i)
    
    return np.array(features), np.array(labels)


def texture_features(img, distances = [4,8,16,32], angles = [0,np.pi/2]):
    """
    Extrait les caractéristiques de texture d'une image.
    
    Parameters:
    -----------
    img : array-like
        Image d'entrée
    distances : list, default=[4,8,16,32]
        Liste des distances pour la matrice de co-occurrence
    angles : list, default=[0, np.pi/2]
        Liste des angles pour la matrice de co-occurrence
    
    Returns:
    --------
    list
        Vecteur de caractéristiques de texture
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (400, 400))
    
    feat_vec = calculate_texture_features(img_resized, distances, angles)
    return feat_vec


def main():
    """
    Fonction principale qui exécute le pipeline de texture:
    - Chargement et prétraitement des images
    - Extraction des caractéristiques de texture
    - Évaluation du classificateur k-NN
    
    Parameters:
    -----------
    None
    
    Returns:
    --------
    None
    """
    # Définition des dossiers contenant les images
    folders = ['pizzafromag', 'pizzahawai', 'pizzamargherita', 'pizzapepperoni', 'pizzareine', 'pizzavege']
    
    # Chemin du dossier principal contenant les images
    base_dir = os.path.join('..', 'masked', 'pizzafromag')
    if not os.path.exists(base_dir):
        base_dir = os.path.join('..', 'masked')
    if not os.path.exists(base_dir):
        base_dir = 'masked'
    
    print(f"Utilisation du dossier de base: {base_dir}")
    
    # Distances pour la matrice de co-occurrence
    distances = [1, 2, 4, 8]
    
    # Charger et prétraiter les images
    features, labels = load_and_preprocess_images(base_dir, folders, distances)
    
    if len(features) == 0:
        print("Aucune image n'a été traitée. Vérifiez les chemins d'accès aux dossiers.")
        exit(1)
    
    print(f"\nNombre total d'images traitées: {len(features)}")
    
    # Séparer les données en entraînement (80%) et test (20%)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    # Tester différentes valeurs de k
    k_values = range(3, 21)
    knn.evaluate_knn(X_train, X_test, y_train, y_test, k_values)

if __name__ == "__main__":
    main()