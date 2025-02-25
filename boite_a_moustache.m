clc; clear; close all;

% Définition des dossiers contenant les images
folders = {'pizzafromag', 'pizzahawai', 'pizzamargherita', 'pizzapepperoni', 'pizzareine', 'pizzavege'};
num_types = numel(folders);

area_data = cell(num_types, 1);
medians = zeros(num_types, 1);
q1 = zeros(num_types, 1);
q3 = zeros(num_types, 1);
mins = zeros(num_types, 1);
maxs = zeros(num_types, 1);

% Parcours des dossiers
for i = 1:num_types
    files = dir(fullfile('Masked_dataset\', folders{i}, '*.jpg')); 
    num_files = numel(files);
    Area = zeros(num_files, 1);
    %lum_values = zeros(num_files, 1);
    
    for j = 1:num_files
        img = imread(fullfile('Masked_dataset\', folders{i}, files(j).name));
        [Area(j)] = caracblanc(img);
        %gray_img = rgb2gray(img);
        %lum_values(j) = area(gray_img(:));
    end
    
    % Stocker les statistiques
    area_data{i} = Area;
    medians(i) = median(Area);
    q1(i) = quantile(Area, 0.25);
    q3(i) = quantile(Area, 0.75);
    mins(i) = min(Area);
    maxs(i) = max(Area);
end

% Affichage des résultats avec scatter et errorbar
figure; hold on;
x = 1:num_types;

% Points individuels
for i = 1:num_types
    scatter(ones(size(area_data{i})) * i, area_data{i}, 'b', 'filled', 'MarkerFaceAlpha', 0.3);
end

% Tracer les médianes
scatter(x, medians, 100, 'r', 'filled');

% Ajouter les barres d'erreur pour min-Q1-Q3-max
errorbar(x, medians, medians - q1, q3 - medians, 'k', 'LineWidth', 2);

% Ajouter les min/max comme lignes verticales
for i = 1:num_types
    plot([i, i], [mins(i), maxs(i)], 'k--', 'LineWidth', 1.5);
end

xticks(x);
xticklabels(folders);
xlabel('Types de Pizza');
ylabel('Aire de la couleur (niveau de gris)');
title('Distribution du vert par Type de Pizza');
hold off;
