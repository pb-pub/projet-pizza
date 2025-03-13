import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imread
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, classification_report  # Ajout de f1_score, precision_score, recall_score, classification_report
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

import cv2
import numpy as np

def sem_algorithm(im, k, max_iter=100):
    """
    Implementation of Stochastic Expectation Maximization for Gaussian Mixture Model

    Args:
        im: Input image (grayscale)
        k: Number of clusters
        max_iter: Maximum number of iterations

    Returns:
        label_img: Labels for each pixel
        params: Parameters of the Gaussian distributions
        log_likelihood: Log-likelihood evolution
    """
    # Initialize random generator
    np.random.seed()

    # Get image dimensions
    dim = im.shape
    label_img = np.zeros(dim)
    size = dim[0] * dim[1]
    params = np.zeros((k, 3))  # mean, variance, prior probability
    posterior_probs = np.zeros((256, k))

    # Initial random labeling
    draw = np.random.rand(*dim)
    initial_law = np.ones(k) / k
    cumsum_law = np.cumsum(initial_law)

    # Initialize parameters
    for ki in range(k):
        if ki == 0:
            indices = draw < cumsum_law[ki]
        else:
            indices = (draw >= cumsum_law[ki-1]) & (draw < cumsum_law[ki])

        label_img[indices] = ki
        im_indices = im[indices]
        params[ki, 0] = np.mean(im_indices.astype(float))  # mean
        params[ki, 1] = np.var(im_indices.astype(float))   # variance
        params[ki, 2] = np.sum(indices) / size             # prior

    # EM iterations
    log_likelihood = np.zeros(max_iter)

    for it in range(max_iter):
        draw = np.random.rand(*dim)

        # For each gray level
        for gray in range(256):
            # Calculate posterior probabilities
            for ki in range(k):
                posterior_probs[gray, ki] = params[ki, 2] * np.exp(
                    -(gray - params[ki, 0])**2 / (2*params[ki, 1])) / np.sqrt(2*np.pi*params[ki, 1])

            # Normalize posteriors
            p_gray = np.sum(posterior_probs[gray])
            if p_gray > 0:
                posterior_probs[gray] /= p_gray

            # Get pixels of current gray level
            pixel_indices = np.where(im == gray)
            n_pixels = len(pixel_indices[0])

            if n_pixels > 0:
                # Sample new labels
                cumsum_post = np.cumsum(posterior_probs[gray])
                pixel_draws = draw[pixel_indices]

                for ki in range(k):
                    if ki == 0:
                        mask = pixel_draws < cumsum_post[ki]
                    else:
                        mask = (pixel_draws >= cumsum_post[ki-1]) & (pixel_draws < cumsum_post[ki])

                    label_img[pixel_indices[0][mask], pixel_indices[1][mask]] = ki

                # Update log-likelihood
                log_likelihood[it] += n_pixels * np.log(p_gray) if p_gray > 0 else 0

        # Update parameters
        for ki in range(k):
            indices = label_img == ki
            if np.any(indices):
                params[ki, 0] = np.mean(im[indices].astype(float))
                params[ki, 1] = np.var(im[indices].astype(float))
                params[ki, 2] = np.sum(indices) / size

    return label_img, params, log_likelihood


def detect_white(img):
    """Détection des pixels blancs"""
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
    """Détection des pixels rouges"""
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
    """Détection des pixels verts"""
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
    """Détection des pixels jaunes"""
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
    """Détection des pixels marrons"""
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
    """Détection des pixels roses"""
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
    """Extraction des caractéristiques de couleur d'une image"""
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
    # Obtient le chemin absolu du script en cours d'exécution
    script_path = os.path.abspath(sys.argv[0])
    # Obtient le répertoire contenant le script
    project_root = os.path.dirname(script_path)
    return project_root

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
    dataset_path = os.path.join(PROJECT_ROOT, 'flori\\Documents\\Cours\\Git\\projet-pizza\masked_dataset_carre', folders[i])
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

# Sauvegarde du fichier Excel dans le répertoire du projet
excel_output_path = os.path.join(PROJECT_ROOT, 'flori\\Documents\\Cours\\Git\\projet-pizza\\output_color.xlsx')
excel_df.to_excel(excel_output_path, index=False)

# Classification sur la couleur & Matrice de confusion correspondante
features = []
labels = []

# Parcours des dossiers
for i in range(num_types):
    dataset_path = os.path.join(PROJECT_ROOT, 'flori\\Documents\\Cours\\Git\\projet-pizza\\masked_dataset_carre', folders[i])
    files = [f for f in os.listdir(dataset_path) if f.endswith('.jpg')]
    for file in files:
        img_path = os.path.join(dataset_path, file)
        img = imread(img_path)
        feat_vec = extract_color_features(img)
        features.append(feat_vec)
        labels.append(i + 1)

# Classification avec SVM
features = np.array(features)
labels = np.array(labels)

# Split des données
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Création du pipeline avec standardisation et SVM
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', C=10, gamma='scale', random_state=42))
])

# Entraînement du modèle
pipeline.fit(X_train, y_train)

# Prédictions
y_pred_train = pipeline.predict(X_train)
y_pred = pipeline.predict(X_test)

# Calcul des métriques
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

# Affichage des résultats
print('\nRésultats avec SVM :')
print('Matrice de confusion :')
print(confusion_matrix(y_test, y_pred))
print(f'\nPrécision sur l\'ensemble d\'entraînement : {train_accuracy:.4f}')
print(f'Précision sur l\'ensemble de test : {test_accuracy:.4f}')
print(f'F1-score : {f1:.4f}')

# Rapport de classification détaillé
print('\nRapport de classification :')
print(classification_report(y_test, y_pred, target_names=folders, zero_division=0))

# Visualisation des décisions
plt.figure(figsize=(12, 6))

# Réduire la dimensionnalité pour la visualisation avec PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Créer une grille pour visualiser les frontières de décision
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Ajuster un nouveau SVM sur les données réduites
svm_2d = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', C=10, gamma='scale', random_state=42))
])
svm_2d.fit(X_train_pca, y_train)

# Prédire sur la grille
Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Tracer les résultats
plt.subplot(121)
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, alpha=0.8)
plt.title("Données d'entraînement")
plt.xlabel("Première composante principale")
plt.ylabel("Deuxième composante principale")

plt.subplot(122)
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, alpha=0.8)
plt.title("Données de test")
plt.xlabel("Première composante principale")
plt.ylabel("Deuxième composante principale")

plt.tight_layout()
plt.show()