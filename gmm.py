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
    # This function should be implemented to extract color features from images
    # Placeholder implementation:
    return np.mean(img, axis=(0, 1))

def evaluate_model(X_train, X_test, y_train, y_test, folders, n_components=None, covariance_type='full'):
    """
    Evaluate a GMM model with the given data.
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
    """Visualize confusion matrix"""
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
    """Load data from the specified folders"""
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