# Classification sur la couleur & Matrice de confusion correspondante
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
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
print(f'\nPrécision sur l\'ensemble d\'entraînement : {train_accuracy:.4f}')
print(f'Précision sur l\'ensemble de test : {test_accuracy:.4f}')
print(f'F1-score : {f1:.4f}')

# Rapport de classification détaillé
print('\nRapport de classification :')
print(classification_report(y_test, y_pred, target_names=folders, zero_division=0))

# Visualisation des matrices de confusion
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Matrice de confusion brute
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=folders,
            yticklabels=folders,
            ax=ax1)
ax1.set_title('Matrice de confusion (valeurs brutes)')
ax1.set_xlabel('Prédiction')
ax1.set_ylabel('Vraie classe')

# Matrice de confusion normalisée
conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
sns.heatmap(conf_matrix_norm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=folders,
            yticklabels=folders,
            ax=ax2)
ax2.set_title('Matrice de confusion (normalisée)')
ax2.set_xlabel('Prédiction')
ax2.set_ylabel('Vraie classe')

plt.tight_layout()
plt.show()

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