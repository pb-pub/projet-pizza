import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.io import imread
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

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
        Vecteur de caractéristiques de couleur (moyenne des canaux)
    """
    # This function should be implemented to extract color features from images
    # Placeholder implementation:
    return np.mean(img, axis=(0, 1))

def evaluate_model(X_train, X_test, y_train, y_test, folders, n_components=None, covariance_type='full'):
    """
    Évalue un modèle GMM avec les données fournies.
    
    Parameters:
    -----------
    X_train, X_test : array-like
        Caractéristiques d'entraînement et de test
    y_train, y_test : array-like
        Étiquettes d'entraînement et de test
    folders : list
        Liste des noms de classes
    n_components : int, optional
        Nombre de composantes gaussiennes par classe
    covariance_type : str, default='full'
        Type de matrice de covariance pour les GMMs
    
    Returns:
    --------
    dict
        Dictionnaire contenant les métriques d'évaluation
    """
    n_classes = len(folders)
    if n_components is None:
        n_components = n_classes
    
    # Créer un GMM par classe
    models = []
    for i in range(n_classes):
        class_samples = X_train[y_train == i]
        gmm = GaussianMixture(
            n_components=2,  # 2 composantes par classe
            covariance_type='full',
            max_iter=200,
            n_init=5,
            reg_covar=1e-3,
            random_state=42+i
        )
        gmm.fit(class_samples)
        models.append(gmm)
    
    # Prédictions
    def predict(X):
        scores = np.array([gmm.score_samples(X) for gmm in models])
        return np.argmax(scores, axis=0)
    
    y_pred_train = predict(X_train)
    y_pred = predict(X_test)
    
    # Calcul des métriques
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Affichage des résultats
    print('\nRésultats avec GMM :')
    print('Matrice de confusion :')
    print(conf_matrix)
    print(f'\nPrécision sur l\'ensemble d\'entraînement : {train_accuracy:.4f}')
    print(f'Précision sur l\'ensemble de test : {test_accuracy:.4f}')
    print(f'F1-score : {f1:.4f}')
    
    # Rapport de classification détaillé
    print('\nRapport de classification :')
    print(classification_report(y_test, y_pred, target_names=folders, 
                              labels=range(len(folders)), zero_division=0))
    
    return {
        'pipeline': models,
        'train_accuracy': train_accuracy, 
        'test_accuracy': test_accuracy,
        'f1_score': f1,
        'confusion_matrix': conf_matrix,
        'y_pred': y_pred,
        'log_likelihood': np.mean([gmm.score(X_test) for gmm in models])
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
            labels.append(i)  # Changed from i + 1 to i
    
    return np.array(features), np.array(labels)

def main():
    """
    Fonction principale qui exécute le pipeline complet:
    - Chargement des données
    - Division en ensembles d'entraînement et de test
    - Évaluation du modèle GMM avec différentes configurations
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
    folders = ['pizzafromag', 'pizzahawai', 'pizzamargherita', 'pizzapepperoni', 'pizzareine', 'pizzavege']
    num_types = len(folders)
    
    # Load data
    features, labels = load_data(PROJECT_ROOT, folders, num_types)
    
    # Split des données
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    # Evaluate model avec différents nombres de composantes par classe
    n_components_per_class_range = range(1, 5)
    results_list = []
    
    for n in n_components_per_class_range:
        print(f"\nTest avec {n} composantes par classe:")
        results = evaluate_model(X_train, X_test, y_train, y_test, folders, n_components=n)
        results_list.append((n, results['test_accuracy'], results['log_likelihood']))
    
    # Afficher les résultats
    plt.figure(figsize=(12, 5))
    
    plt.subplot(121)
    plt.plot([r[0] for r in results_list], [r[1] for r in results_list], 'o-')
    plt.xlabel('Nombre de composantes par classe')
    plt.ylabel('Précision sur le test')
    plt.title('Précision en fonction du nombre de composantes par classe')
    
    plt.subplot(122)
    plt.plot([r[0] for r in results_list], [r[2] for r in results_list], 'o-')
    plt.xlabel('Nombre de composantes par classe')
    plt.ylabel('Log-vraisemblance')
    plt.title('Log-vraisemblance en fonction du nombre de composantes par classe')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()