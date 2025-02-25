function [AreaI] = caracrouge(I)
% Convertir en espace HSV
Ihsv = rgb2hsv(I);

% Extraire les canaux HSV
H = Ihsv(:,:,1); % Teinte (Hue)
S = Ihsv(:,:,2); % Saturation
V = Ihsv(:,:,3); % Valeur (Brightness)

% Définir les seuils pour le vert (ajuster si besoin)
H_min1 = 0.0; H_max1 = 0.06; % Rouge vif (0° à 21.6°)
H_min2 = 0.94; H_max2 = 1.0; % Rouge foncé (338.4° à 360°)
S_min = 0.5; % Couleur bien saturée
V_min = 0.3; % Inclure les rouges moyens et vifs


% Créer un masque binaire pour le jaune
mask = ((H >= H_min1) & (H <= H_max1) | (H >= H_min2) & (H <= H_max2)) & (S >= S_min) & (V >= V_min);

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
