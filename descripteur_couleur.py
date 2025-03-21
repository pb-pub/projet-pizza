import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imread
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import knn
import cv2
import numpy as np

def detect_white(img):
    """
    Détecte les pixels blancs dans une image.
    
    Parameters:
    -----------
    img : array-like
        Image d'entrée
    
    Returns:
    --------
    float
        Nombre de pixels détectés comme blancs
    """
    # Convertir l'image en uint8 si nécessaire
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Seuils pour le blanc
    s_max = int(0.2 * 255)  # Faible saturation
    v_min = int(0.7 * 255)  # Haute luminosité

    mask = (hsv[:,:,1] <= s_max) & (hsv[:,:,2] >= v_min)
    mask = mask.astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return float(np.sum(mask))

def detect_red(img):
    """
    Détecte les pixels rouges dans une image.
    
    Parameters:
    -----------
    img : array-like
        Image d'entrée
    
    Returns:
    --------
    float
        Nombre de pixels détectés comme rouges
    """
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Seuils pour le rouge (deux plages en HSV)
    lower_red1 = np.array([0, int(0.5*255), int(0.3*255)], dtype=np.uint8)
    upper_red1 = np.array([int(0.06*180), 255, 255], dtype=np.uint8)
    lower_red2 = np.array([int(0.94*180), int(0.5*255), int(0.3*255)], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.add(mask1, mask2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return float(np.sum(mask > 0))

def detect_green(img):
    """
    Détecte les pixels verts dans une image.
    
    Parameters:
    -----------
    img : array-like
        Image d'entrée
    
    Returns:
    --------
    float
        Nombre de pixels détectés comme verts
    """
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    lower_green = np.array([int(0.15*180), int(0.3*255), int(0.2*255)], dtype=np.uint8)
    upper_green = np.array([int(0.55*180), 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return float(np.sum(mask > 0))

def detect_yellow(img):
    """
    Détecte les pixels jaunes dans une image.
    
    Parameters:
    -----------
    img : array-like
        Image d'entrée
    
    Returns:
    --------
    float
        Nombre de pixels détectés comme jaunes
    """
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    lower_yellow = np.array([0.10*180, 0.4*255, 0.5*255], dtype=np.uint8)
    upper_yellow = np.array([0.18*180, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return float(np.sum(mask > 0))

def detect_brown(img):
    """
    Détecte les pixels marrons dans une image.
    
    Parameters:
    -----------
    img : array-like
        Image d'entrée
    
    Returns:
    --------
    float
        Nombre de pixels détectés comme marrons
    """
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    lower_brown = np.array([0.02*180, 0.3*255, 0.2*255], dtype=np.uint8)
    upper_brown = np.array([0.08*180, 0.8*255, 0.6*255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_brown, upper_brown)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return float(np.sum(mask > 0))

def detect_pink(img):
    """
    Détecte les pixels roses dans une image.
    
    Parameters:
    -----------
    img : array-like
        Image d'entrée
    
    Returns:
    --------
    float
        Nombre de pixels détectés comme roses
    """
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    lower_pink = np.array([0.8*180, 0.2*255, 0.7*255], dtype=np.uint8)
    upper_pink = np.array([0.95*180, 0.6*255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_pink, upper_pink)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return float(np.sum(mask > 0))

def extract_color_features(img):
    """
    Extraction des caractéristiques de couleur d'une image.
    
    Parameters:
    -----------
    img : array-like
        Image d'entrée
    
    Returns:
    --------
    array
        Vecteur contenant les proportions de chaque couleur
        (blanc, rouge, vert, jaune, marron, rose)
    """
    img_size = float(img.shape[0] * img.shape[1])

    return np.array([
        detect_white(img) / img_size,    # Blanc
        detect_red(img) / img_size,      # Rouge
        detect_green(img) / img_size,    # Vert
        detect_yellow(img) / img_size,   # Jaune
        detect_brown(img) / img_size,    # Marron
        detect_pink(img) / img_size      # Rose
    ])

def get_project_root():
    """
    Obtient le chemin racine du projet.
    
    Parameters:
    -----------
    None
    
    Returns:
    --------
    str
        Chemin absolu du répertoire du projet
    """
    # Obtient le chemin absolu du script en cours d'exécution
    script_path = os.path.abspath(sys.argv[0])
    # Obtient le répertoire contenant le script
    project_root = os.path.dirname(script_path)
    return project_root


if __name__ == '__main__':
    # Chemin racine du projet
    PROJECT_ROOT = get_project_root()

    # Définition des dossiers contenant les images
    folders = ['pizzafromag', 'pizzahawai', 'pizzamargherita', 'pizzapepperoni', 'pizzareine', 'pizzavege']
    num_types = len(folders)

    area_data = [None] * num_types
    medians = np.zeros((num_types, 6))
    q1 = np.zeros((num_types, 6))
    q3 = np.zeros((num_types, 6))
    mins = np.zeros((num_types, 6))
    maxs = np.zeros((num_types, 6))

    # Parcours des dossiers
    for i in range(num_types):
        dataset_path = os.path.join(PROJECT_ROOT, '../masked', folders[i])
        files = [f for f in os.listdir(dataset_path) if f.endswith('.jpg')]
        num_files = len(files)
        Areas = np.zeros((num_files, 6))

        for j, file in enumerate(files):
            img_path = os.path.join(dataset_path, file)
            img = imread(img_path)
            Areas[j, :] = extract_color_features(img)

        # Stocker les statistiques
        area_data[i] = Areas
        medians[i, :] = np.median(Areas, axis=0)
        q1[i, :] = np.percentile(Areas, 25, axis=0)
        q3[i, :] = np.percentile(Areas, 75, axis=0)
        mins[i, :] = np.min(Areas, axis=0)
        maxs[i, :] = np.max(Areas, axis=0)

    # Affichage des boîtes à moustaches
    fig, axs = plt.subplots(3, 2, figsize=(15, 10))
    colors = ['Blanc', 'Rouge', 'Vert', 'Jaune', 'Marron', 'Rose']

    for c in range(6):
        ax = axs[c // 2, c % 2]
        ax.set_title(f'Distribution du {colors[c]} par Type de Pizza')
        ax.set_xlabel('Types de Pizza')
        ax.set_ylabel('Aire de la couleur (niveau de gris)')
        x = np.arange(1, num_types + 1)

        for i in range(num_types):
            ax.scatter(np.ones(area_data[i].shape[0]) * (i + 1), area_data[i][:, c], color='b', alpha=0.3)

        ax.scatter(x, medians[:, c], color='r', s=100)
        ax.errorbar(x, medians[:, c], yerr=[medians[:, c] - q1[:, c], q3[:, c] - medians[:, c]], fmt='k', linewidth=2)

        for i in range(num_types):
            ax.plot([i + 1, i + 1], [mins[i, c], maxs[i, c]], 'k--', linewidth=1.5)

        ax.set_xticks(x)
        ax.set_xticklabels(folders)

    plt.tight_layout()
    plt.show()

    # Convertir en table avec des noms de colonnes uniques
    excel_data = []
    for i in range(num_types):
        for j in range(6):
            for k, value in enumerate(area_data[i][:, j]):
                excel_data.append([i + 1, j + 1, k + 1, value])

    excel_df = pd.DataFrame(excel_data, columns=['RowIdx', 'ColumnIdx', 'ElementIdx', 'Value'])

    # Classification sur la couleur & Matrice de confusion correspondante
    features = []
    labels = []

    # Parcours des dossiers
    for i in range(num_types):
        dataset_path = os.path.join(PROJECT_ROOT, '../masked', folders[i])
        files = [f for f in os.listdir(dataset_path) if f.endswith('.jpg')]
        for file in files:
            img_path = os.path.join(dataset_path, file)
            img = imread(img_path)
            feat_vec = extract_color_features(img)
            features.append(feat_vec)
            labels.append(i + 1)

    features = np.array(features)
    labels = np.array(labels)
    # save the features in a file
    np.save('features_flo.npy', features)
    np.save('labels_flo.npy', labels)
    

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    

    knn.evaluate_knn(X_train, X_test, y_train, y_test, [6])