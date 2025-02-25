clc; clear; close all;

% Lire l'image
I = imread(['Masked_dataset/pizzafromag/f30.jpg']); % Remplace par ton image

% Convertir en espace HSV
Ihsv = rgb2hsv(I);

% Extraire les canaux HSV
H = Ihsv(:,:,1); % Teinte (Hue)
S = Ihsv(:,:,2); % Saturation
V = Ihsv(:,:,3); % Valeur (Brightness)

% Définir les seuils pour détecter le blanc (fromage de chèvre)
S_max = 0.2;  % Faible saturation (évite les couleurs vives)
V_min = 0.7;  % Haute luminosité (cible les zones blanches)

% Créer un masque binaire pour le blanc
mask = (S <= S_max) & (V >= V_min);

% Appliquer une ouverture morphologique pour réduire le bruit
se = strel('disk', 3);
mask_clean = imopen(mask, se);

% Afficher les résultats
figure;
subplot(1,3,1);
imshow(I);
title('Image originale');

subplot(1,3,2);
imshow(mask);
title('Masque initial (Seuillage)');

subplot(1,3,3);
imshow(mask_clean);
title('Masque après traitement');
