function feat_vec = caraccouleur(img)

img_size = size(img, 1) * size(img, 2);

feat_vec = [
    caracblanc(img) / img_size;   % Blanc
    caracrouge(img) / img_size;   % Rouge
    caracvert(img) / img_size;    % Vert
    caracjaune(img) / img_size;  % Jaune
    caracmarron(img) / img_size; % Marron
    caracrose(img) / img_size;   % Rose
];

end
