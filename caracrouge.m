function [AreaI] = caracrouge(I)
% Convertir en espace HSV
Ihsv = rgb2hsv(I);

% Extraire les canaux HSV
H = Ihsv(:,:,1); % Teinte (Hue)
S = Ihsv(:,:,2); % Saturation
V = Ihsv(:,:,3); % Valeur (Brightness)

% Définir les seuils pour le vert (ajuster si besoin)
H_min = 0.9; % Approximativement jaune (~30° en HSV)
H_max = 1; 
S_min = 0.2;  % Pour éviter les zones blanches
S_max = 0.5;  % Pour éviter les zones blanches
V_min = 0.3;  % Exclure les parties sombres

% Créer un masque binaire pour le jaune
mask = (H >= H_min) & (H <= H_max) & (S >= S_min) & (S <= S_max) & (V >= V_min);

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
