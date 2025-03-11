import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score

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
