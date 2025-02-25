function [AreaI] = caracmarron(I)
% Convertir en espace HSV
Ihsv = rgb2hsv(I);

% Extraire les canaux HSV
H = Ihsv(:,:,1); % Teinte (Hue)
S = Ihsv(:,:,2); % Saturation
V = Ihsv(:,:,3); % Valeur (Brightness)

% Définir les seuils pour le marron/gris
H_min = 0.08; % Environ 30° (beige clair)
H_max = 0.15; % Environ 54° (marron clair)
S_min = 0.1;  % Peu saturé
S_max = 0.4;  % Pas trop de couleur vive
V_min = 0.3;  % Éviter les zones trop sombres
V_max = 0.8;  % Éviter les blancs trop lumineux

% Créer un masque binaire pour le jaune
mask = (H >= H_min) & (H <= H_max) & (S >= S_min) & (S <= S_max) & (V >= V_min) & (V <= V_max);

% Appliquer une ouverture morphologique pour réduire le bruit
se = strel('disk', 3);
Ivert = imopen(mask, se);
AreaI = sum(sum(Ivert,2),1);

end

% % Afficher les résultats
% figure;
% subplot(1,3,1);
% imshow(I);
% title('Image originale');
% 
% subplot(1,3,2);
% imshow(mask);
% title('Masque initial (Seuillage)');
% 
% subplot(1,3,3);
% imshow(mask_clean);
% title('Masque après traitement');
