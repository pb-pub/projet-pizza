import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.io import imread
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA

def extract_color_features(img):
    """
    Extrait les caractéristiques de couleur d'une image.
    
    Parameters:
    -----------
    img : array-like
        Image d'entrée
    
    Returns:
    --------
    array
        Vecteur de caractéristiques de couleur
    """
    # This function should be implemented to extract color features from images
    # Placeholder implementation:
    return np.mean(img, axis=(0, 1))

def evaluate_model(X_train, X_test, y_train, y_test, folders, C=10, gamma='scale', kernel='rbf'):
    """
    Évalue un modèle SVM avec les données fournies.
    
    Parameters:
    -----------
    X_train, X_test : array-like
        Caractéristiques d'entraînement et de test
    y_train, y_test : array-like
        Étiquettes d'entraînement et de test
    folders : list
        Liste des noms de classes
    C, gamma, kernel : paramètres SVM
        Paramètres de configuration du modèle SVM
    
    Returns:
    --------
    dict
        Dictionnaire contenant les métriques d'évaluation
    """
    # Création du pipeline avec standardisation et SVM
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel=kernel, C=C, gamma=gamma, random_state=42))
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
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Affichage des résultats
    print('\nRésultats avec SVM :')
    print('Matrice de confusion :')
    print(conf_matrix)
    print(f'\nPrécision sur l\'ensemble d\'entraînement : {train_accuracy:.4f}')
    print(f'Précision sur l\'ensemble de test : {test_accuracy:.4f}')
    print(f'F1-score : {f1:.4f}')
    
    # Rapport de classification détaillé
    print('\nRapport de classification :')
    print(classification_report(y_test, y_pred, target_names=folders, zero_division=0))
    
    return {
        'pipeline': pipeline,
        'train_accuracy': train_accuracy, 
        'test_accuracy': test_accuracy,
        'f1_score': f1,
        'confusion_matrix': conf_matrix,
        'y_pred': y_pred
    }

def visualize_confusion_matrix(conf_matrix, folders):
    """
    Visualise la matrice de confusion.
    
    Parameters:
    -----------
    conf_matrix : array-like
        Matrice de confusion à visualiser
    folders : list
        Liste des noms de classes
    
    Returns:
    --------
    None
        Affiche les graphiques de la matrice de confusion
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Matrice de confusion brute
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Reds',
                xticklabels=folders,
                yticklabels=folders,
                ax=ax1)
    ax1.set_title('Matrice de confusion (valeurs brutes)')
    ax1.set_xlabel('Prédiction')
    ax1.set_ylabel('Vraie classe')

    # Matrice de confusion normalisée
    conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    sns.heatmap(conf_matrix_norm, annot=True, fmt='.2f', cmap='Reds',
                xticklabels=folders,
                yticklabels=folders,
                ax=ax2)
    ax2.set_title('Matrice de confusion (normalisée)')
    ax2.set_xlabel('Prédiction')
    ax2.set_ylabel('Vraie classe')

    plt.tight_layout()
    plt.show()

def visualize_decision_boundaries(X_train, X_test, y_train, y_test):
    """
    Visualise les frontières de décision à l'aide de PCA.
    
    Parameters:
    -----------
    X_train, X_test : array-like
        Caractéristiques d'entraînement et de test
    y_train, y_test : array-like
        Étiquettes d'entraînement et de test
    
    Returns:
    --------
    None
        Affiche les graphiques des frontières de décision
    """
    plt.figure(figsize=(12, 6))

    # Réduire la dimensionnalité pour la visualisation avec PCA
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

def load_data(PROJECT_ROOT, folders, num_types):
    """
    Charge les données depuis les dossiers spécifiés.
    
    Parameters:
    -----------
    PROJECT_ROOT : str
        Chemin racine du projet
    folders : list
        Liste des dossiers contenant les images
    num_types : int
        Nombre de types de pizzas
    
    Returns:
    --------
    tuple
        Tuple contenant les caractéristiques et les étiquettes (features, labels)
    """
    features = []
    labels = []

    # Parcours des dossiers
    for i in range(num_types):
        dataset_path = os.path.join(PROJECT_ROOT, 'Documents\\Cours\\Git\\projet-pizza\\masked_dataset_carre', folders[i])
        files = [f for f in os.listdir(dataset_path) if f.endswith('.jpg')]
        for file in files:
            img_path = os.path.join(dataset_path, file)
            img = imread(img_path)
            feat_vec = extract_color_features(img)
            features.append(feat_vec)
            labels.append(i + 1)
    
    return np.array(features), np.array(labels)

def main():
    """
    Fonction principale qui exécute le pipeline complet:
    - Chargement des données
    - Division en ensembles d'entraînement et de test
    - Évaluation du modèle SVM
    - Visualisation des résultats
    
    Parameters:
    -----------
    None
    
    Returns:
    --------
    None
    """
    # Define constants
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    folders = ['calzone', 'margherita', 'marinara', 'pugliese']
    num_types = len(folders)
    
    # Load data
    features, labels = load_data(PROJECT_ROOT, folders, num_types)
    
    # Split des données
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    # Evaluate model
    results = evaluate_model(X_train, X_test, y_train, y_test, folders)
    
    # Visualize results
    visualize_confusion_matrix(results['confusion_matrix'], folders)
    visualize_decision_boundaries(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()