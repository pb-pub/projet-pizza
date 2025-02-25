function [AreaI] = caracjaune(I)
% Convertir en espace HSV
Ihsv = rgb2hsv(I);

% Extraire les canaux HSV
H = Ihsv(:,:,1); % Teinte (Hue)
S = Ihsv(:,:,2); % Saturation
V = Ihsv(:,:,3); % Valeur (Brightness)

% Définir les seuils pour détecter le jaune (fromage de chèvre)
H_min = 0.10; % Approximativement jaune (~30° en HSV)
H_max = 0.18; 
S_min = 0.4;  % Pour éviter les zones blanches
V_min = 0.5;  % Exclure les parties sombres

% Créer un masque binaire pour le blanc
mask = (H >= H_min) & (H <= H_max) & (S >= S_min) & (V >= V_min);

% Appliquer une ouverture morphologique pour réduire le bruit
se = strel('disk', 3);
Ijaune = imopen(mask, se);
AreaI = sum(sum(Ijaune,2),1);
% Afficher les résultats
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
% imshow(Iblanc);
% title('Masque après traitement');

end