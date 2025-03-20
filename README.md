# projet-pizza

## Overview
Ce projet implémente une chaîne de traitement et de classification d'images de pizza. L'objectif est de détecter, segmenter et extraire des caractéristiques (texture et couleur) à partir d'images de pizzas, puis de classer ces images à l'aide d'un algorithme k-NN.

## Project Structure
- **main.py** : Point d'entrée pour l'extraction de caractéristiques et la classification.
- **knn.py** : Fonctions d'évaluation et de classification k-NN.
- **hough.py** : Détection et masquage de la zone de pizza par transformation de Hough.
- **descripteurs_texture.py** et **descripteur_couleur.py** : Extraction des caractéristiques de texture et de couleur.

## Setup & Dependencies
- Python 3.x
- OpenCV
- NumPy
- scikit-learn
- Seaborn
- Matplotlib
 
## How to Run
1. Placez vos images de pizza dans le dossier `dataset` ou `masked` selon le prétraitement.
2. Exécutez **main.py** pour extraire les caractéristiques et classifier les images.
3. Utilisez la GUI de **classement_bdd.py** pour trier et vérifier manuellement les images.

## Descripteurs

### Descripteur Couleur
Les caractéristiques de couleur sont extraites dans le fichier `descripteur_couleur.py`. Pour chaque image, plusieurs indicateurs sont calculés à partir de l'espace HSV :
- **Blanc** : Proportion de pixels à faible saturation et haute luminosité.
- **Rouge** : Détecte le rouge via deux plages de valeurs HSV afin de couvrir la circularité de l'espace de teinte.
- **Vert** : Mesure les pixels verts dans l'image.
- **Jaune** : Détecte la présence de pixels jaunes.
- **Marron** : Estime la présence de pixels marrons.
- **Rose** : Mesure la proportion de pixels roses.

Ces ratios, rapportés à la taille de l'image, constituent le vecteur de caractéristiques couleur.

### Descripteur Texture
Les caractéristiques de texture sont extraites dans le fichier `descripteurs_texture.py`. L'approche utilisée repose sur le calcul de la matrice de cooccurrence en niveaux de gris (GLCM) :
- Pour chaque image convertie en niveaux de gris et redimensionnée, la GLCM est calculée pour plusieurs distances et orientations.
- Deux propriétés principales sont extraites à partir de la GLCM :
  - **Corrélation** : Mesure la linéarité des relations entre les niveaux de gris.
  - **Énergie** : Mesure l'homogénéité de la texture.

Le calcul des descripteurs de texture s'appuie sur la Matrice de Co-occurrence des Niveaux de Gris (GLCM) afin de quantifier l'aspect textural des images de pizza. Voici comment le processus est réalisé :

1. **Prétraitement de l'image**  
   Dans le fichier **descripteurs_texture.py**, l'image est d'abord convertie en niveaux de gris et redimensionnée (par exemple à 400x400) pour homogénéiser le calcul des caractéristiques.

2. **Paramètres de la GLCM**  
   - **Distances** : Une liste de distances, typiquement `[1, 2, 4, 8]`, est utilisée pour mesurer les relations entre pixels situés à différentes séparations. Ces distances permettent de capter des détails de texture à diverses échelles.
   - **Angles** : Des orientations, comme `[0, π/4, π/2, 3π/4]` (ou dans certains cas simplifiés `[0, π/2]`), sont considérées afin de mesurer la texture dans plusieurs directions. Cela permet d'obtenir une représentation robuste malgré les variations directionnelles dans l'image.
   - **Levels** : La GLCM est calculée avec `levels=256`, assurant ainsi la prise en compte de tous les niveaux de gris disponibles dans l'image.
   - **Symétrie et Normalisation** : Les options `symmetric=True` et `normed=True` garantissent respectivement que la matrice est symétrique et que ses valeurs sont normalisées, ce qui facilite la comparaison entre images.

3. **Extraction des Propriétés**  
   Pour chaque combinaison distance/angle, deux propriétés principales sont extraites à l'aide de la fonction `graycoprops` :
   - **Corrélation** : Cette propriété quantifie la linéarité entre pixels voisins. Elle permet d'évaluer la dépendance linéaire et donc la régularité de la texture.
   - **Énergie** : Aussi appelée uniformité, elle mesure l'homogénéité de la distribution des niveaux de gris. Une valeur élevée indique une texture plus uniforme.

   Pour chaque distance, la moyenne des valeurs obtenues sur les différentes orientations est calculée, et ces moyennes (de corrélation et d'énergie) sont concaténées pour former le vecteur de caractéristiques final.

4. **Utilisation dans le Pipeline de Traitement**  
   Dans le fichier **main.py**, après prétraitement de l'image (masquage via Hough dans `hough.mask_pizza`), la fonction `texture_features` du module `descripteurs_texture` est appelée pour extraire ces caractéristiques. Ces descripteurs de texture sont ensuite concaténés avec les descripteurs de couleur pour former le vecteur d'entrée du classifieur k-NN.

Ces paramètres et étapes assurent que les différences de texture, souvent subtiles dans les images de pizza, soient efficacement quantifiées et utilisées pour la classification.

## Authors
Raphaël BACK - Paul BOUTET - Pierre BOURGEY - Florian GIURGIU🤌 - Alexia LACAN
