clc; clear; close all;

%% Boite à moustache : Blanc, Rouge, Vert, Jaune et Marron

% Définition des dossiers contenant les images
folders = {'pizzafromag', 'pizzahawai', 'pizzamargherita', 'pizzapepperoni', 'pizzareine', 'pizzavege'};
num_types = numel(folders);

area_data = cell(num_types, 5);
medians = zeros(num_types, 5);
q1 = zeros(num_types, 5);
q3 = zeros(num_types, 5);
mins = zeros(num_types, 5);
maxs = zeros(num_types, 5);

% Parcours des dossiers
for i = 1:num_types
    files = dir(fullfile('masked_dataset\', folders{i}, '*.jpg')); 
    num_files = numel(files);
    Areas = zeros(num_files, 5);
    
    for j = 1:num_files
        img = imread(fullfile('masked_dataset\', folders{i}, files(j).name));
        Areas(j, 1) = caracblanc(img) / (size(img,1) * size(img,2));
        Areas(j, 2) = caracrouge(img) / (size(img,1) * size(img,2));
        Areas(j, 3) = caracvert(img) / (size(img,1) * size(img,2));
        Areas(j, 4) = caracjaune(img) / (size(img,1) * size(img,2));
        Areas(j, 5) = caracmarron(img) / (size(img,1) * size(img,2));
    end
    
    % Stocker les statistiques
    area_data(i, :) = num2cell(Areas, 1);
    medians(i, :) = median(Areas);
    q1(i, :) = quantile(Areas, 0.25);
    q3(i, :) = quantile(Areas, 0.75);
    mins(i, :) = min(Areas);
    maxs(i, :) = max(Areas);
end

% Affichage des boîtes à moustaches
figure;
colors = {'Blanc', 'Rouge', 'Vert', 'Jaune', 'Marron'};

for c = 1:5
    subplot(3,2,c); hold on;
    x = 1:num_types;
    
    % Points individuels
    for i = 1:num_types
        scatter(ones(size(area_data{i, c})) * i, area_data{i, c}, 'b', 'filled', 'MarkerFaceAlpha', 0.3);
    end
    
    % Tracer les médianes
    scatter(x, medians(:, c), 100, 'r', 'filled');
    
    % Ajouter les barres d'erreur pour min-Q1-Q3-max
    errorbar(x, medians(:, c), medians(:, c) - q1(:, c), q3(:, c) - medians(:, c), 'k', 'LineWidth', 2);
    
    % Ajouter les min/max comme lignes verticales
    for i = 1:num_types
        plot([i, i], [mins(i, c), maxs(i, c)], 'k--', 'LineWidth', 1.5);
    end
    
    xticks(x);
    xticklabels(folders);
    xlabel('Types de Pizza');
    ylabel('Aire de la couleur (niveau de gris)');
    title(['Distribution du ', colors{c}, ' par Type de Pizza']);
    hold off;
end
