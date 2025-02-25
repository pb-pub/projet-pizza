import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import glob
from skimage.feature import graycomatrix, graycoprops

# Définition des dossiers contenant les images
folders = ['pizzafromag', 'pizzahawai', 'pizzamargherita', 'pizzapepperoni', 'pizzareine', 'pizzavege']
num_types = len(folders)

# Chemin du dossier principal contenant les images
base_dir = os.path.join('..', 'masked', 'pizzafromag')  # On remonte d'un niveau depuis projet-pizza
if not os.path.exists(base_dir):
    base_dir = os.path.join('..', 'masked')
if not os.path.exists(base_dir):
    base_dir = 'masked'

print(f"Utilisation du dossier de base: {base_dir}")

# Distances pour la matrice de co-occurrence
distances = [1, 2, 4, 8]

# Stockage des caractéristiques et des labels
features = []
labels = []

for i, folder in enumerate(folders, 1):
    # Construire le chemin complet du dossier
    folder_path = os.path.join(base_dir, folder)
    print(f"Traitement du dossier: {folder_path}")
    
    # Utiliser glob pour lister tous les fichiers jpg
    files = glob.glob(os.path.join(folder_path, '*.jpg'))
    print(f"Nombre d'images trouvées dans {folder}: {len(files)}")
    
    for file in files:
        # Charger l'image
        img = cv2.imread(file)
        if img is None:
            print(f"Impossible de charger l'image: {file}")
            continue
            
        # Convertir en niveaux de gris
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Créer le masque (supposant que le fond est noir)
        mask = img_gray > 0
        
        # Redimensionner à 400x400
        img_resized = cv2.resize(img_gray, (400, 400))
        
        # Masque mis à jour après redimensionnement
        mask_resized = img_resized > 0
        
        # Initialiser un vecteur de caractéristiques pour cette image
        feat_vec = []
        
        # Calculer la GLCM pour chaque distance
        for d in distances:
            # Calculer la GLCM pour différentes orientations
            angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
            glcm = graycomatrix(img_resized, 
                              distances=[d], 
                              angles=angles,
                              levels=256,
                              symmetric=True, 
                              normed=True)
            
            # Calculer les propriétés de texture
            contrast = graycoprops(glcm, 'contrast').mean()
            correlation = graycoprops(glcm, 'correlation').mean()
            energy = graycoprops(glcm, 'energy').mean()
            homogeneity = graycoprops(glcm, 'homogeneity').mean()
            
            # Ajouter à la feature vector
            feat_vec.extend([contrast, correlation, energy, homogeneity])
        
        # Stocker les caractéristiques et le label
        features.append(feat_vec)
        labels.append(i)  # Label numérique correspondant à la pizza

# Vérifier qu'on a des données
if len(features) == 0:
    print("Aucune image n'a été traitée. Vérifiez les chemins d'accès aux dossiers.")
    exit(1)

print(f"\nNombre total d'images traitées: {len(features)}")

# Convertir en array numpy
features = np.array(features)
labels = np.array(labels)

# Séparer les données en entraînement (80%) et test (20%)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Tester différentes valeurs de k
for k in range(3, 21):
    # Création du modèle k-NN
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)
    
    # Prédiction sur l'ensemble de test
    y_pred = knn_model.predict(X_test)
    
    # Calculer la matrice de confusion
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Calculer la précision
    accuracy = np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix)
    
    # Afficher les résultats
    print(f'k : {k}')
    print('Matrice de confusion :')
    print(conf_matrix)
    print(f'Précision globale : {accuracy * 100:.2f}%\n')