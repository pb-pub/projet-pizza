import sys
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt

# Set default encoding to UTF-8
sys.stdout.reconfigure(encoding='utf-8')

def evaluate_knn(X_train, X_test, y_train, y_test, k_values):
    """Évalue le modèle k-NN pour différentes valeurs de k."""
    for k in k_values:
        knn_model = KNeighborsClassifier(n_neighbors=k)
        knn_model.fit(X_train, y_train)
        
        y_pred = knn_model.predict(X_test)
        conf_matrix = confusion_matrix(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)

        print(f'k : {k}')
        print('Matrice de confusion :')
        print(conf_matrix)
        print(f'Précision globale : {accuracy * 100:.2f}%\n')
        
        if k == 15:
            folders = ['pizzafromag', 'pizzahawai', 'pizzamargherita', 'pizzapepperoni', 'pizzareine', 'pizzavege']
            
            # affichage de la matrice de confusion avec sns
            # Visualisation des matrices de confusion
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

            # Matrice de confusion brute
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                        xticklabels=folders,
                        yticklabels=folders,
                        ax=ax1)
            ax1.set_title(f'Matrice de confusion (valeurs brutes), k = {k}, Précision globale : {accuracy * 100:.2f}%')
            ax1.set_xlabel('Prédiction')
            ax1.set_ylabel('Vraie classe')

            # Matrice de confusion normalisée
            conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
            sns.heatmap(conf_matrix_norm, annot=True, fmt='.2f', cmap='Blues',
                        xticklabels=folders,
                        yticklabels=folders,
                        ax=ax2)
            ax2.set_title(f'Matrice de confusion (normalisées par lignes), k = {k}, Précision globale : {accuracy * 100:.2f}%')
            ax2.set_xlabel('Prédiction')
            ax2.set_ylabel('Vraie classe')

            plt.tight_layout()
            plt.show()

            # Visualisation des décisions
            plt.figure(figsize=(12, 6))
            
            
            # Calcul des métriques
            conf_matrix = confusion_matrix(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Affichage des métriques
            print("\nMétriques d'évaluation :")
            print(f'Précision globale (accuracy) : {accuracy:.4f}')
            print(f'Précision pondérée (precision) : {precision:.4f}')
            print(f'Rappel pondéré (recall) : {recall:.4f}')
            print(f'F1-score pondéré : {f1:.4f}')

            # Rapport de classification détaillé
            print("\nRapport de classification détaillé :")
            print(classification_report(y_test, y_pred, target_names=folders, zero_division=0))




