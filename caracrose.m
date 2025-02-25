function [AreaI] = caracmarron(I)
% Convertir en espace HSV
Ihsv = rgb2hsv(I);

% Extraire les canaux HSV
H = Ihsv(:,:,1); % Teinte (Hue)
S = Ihsv(:,:,2); % Saturation
V = Ihsv(:,:,3); % Valeur (Brightness)

% Définir les seuils pour le rose
H_min1 = 0.94; % ≈ 338° (rose clair)
H_max1 = 1.0;  % ≈ 360° (rouge clair)
H_min2 = 0.0;  % ≈ 0° (rouge)
H_max2 = 0.04; % ≈ 14° (rose foncé)
S_min = 0.2;   % Modérément saturé
S_max = 0.6;   % Pas trop vif
V_min = 0.6;   % Lumineux
V_max = 1.0;   % Pas trop foncé

% Créer un masque binaire pour le jambon (2 parties à cause du cycle HSV)
mask1 = (H >= H_min1) & (H <= H_max1) & (S >= S_min) & (S <= S_max) & (V >= V_min) & (V <= V_max);
mask2 = (H >= H_min2) & (H <= H_max2) & (S >= S_min) & (S <= S_max) & (V >= V_min) & (V <= V_max);
mask = mask1 | mask2;

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