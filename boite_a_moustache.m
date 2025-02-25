clc; clear; close all;

%% Boite à moustache : Blanc, Rouge, Vert et Jaune

% Définition des dossiers contenant les images
folders = {'pizzafromag', 'pizzahawai', 'pizzamargherita', 'pizzapepperoni', 'pizzareine', 'pizzavege'};
num_types = numel(folders);

area_data_blanc = cell(num_types, 1);
medians_blanc = zeros(num_types, 1);
q1_blanc = zeros(num_types, 1);
q3_blanc = zeros(num_types, 1);
mins_blanc = zeros(num_types, 1);
maxs_blanc = zeros(num_types, 1);

area_data_rouge = cell(num_types, 1);
medians_rouge = zeros(num_types, 1);
q1_rouge = zeros(num_types, 1);
q3_rouge = zeros(num_types, 1);
mins_rouge = zeros(num_types, 1);
maxs_rouge = zeros(num_types, 1);

area_data_vert = cell(num_types, 1);
medians_vert = zeros(num_types, 1);
q1_vert = zeros(num_types, 1);
q3_vert = zeros(num_types, 1);
mins_vert = zeros(num_types, 1);
maxs_vert = zeros(num_types, 1);

area_data_jaune = cell(num_types, 1);
medians_jaune = zeros(num_types, 1);
q1_jaune = zeros(num_types, 1);
q3_jaune = zeros(num_types, 1);
mins_jaune = zeros(num_types, 1);
maxs_jaune = zeros(num_types, 1);

% Parcours des dossiers
for i = 1:num_types
    files = dir(fullfile('masked_dataset\', folders{i}, '*.jpg')); 
    num_files = numel(files);
    AreaBlanc = zeros(num_files, 1);
    AreaRouge = zeros(num_files, 1);
    AreaVert = zeros(num_files, 1);
    % LumValues = zeros(num_files, 1);
    
    for j = 1:num_files
        img = imread(fullfile('masked_dataset\', folders{i}, files(j).name));
        % [LumValues(j)] = (img(:,:,1)+img(:,:,2)+img(:,:,3))/3;
        [AreaBlanc(j)] = caracblanc(img)/(size(img,1)*size(img,2));
        [AreaRouge(j)] = caracrouge(img)/(size(img,1)*size(img,2));
        [AreaVert(j)] = caracvert(img)/(size(img,1)*size(img,2));
        [AreaJaune(j)] = caracjaune(img)/(size(img,1)*size(img,2));
    end
    
    % Stocker les statistiques
    area_data_blanc{i} = AreaBlanc;
    medians_blanc(i) = median(AreaBlanc);
    q1_blanc(i) = quantile(AreaBlanc, 0.25);
    q3_blanc(i) = quantile(AreaBlanc, 0.75);
    mins_blanc(i) = min(AreaBlanc);
    maxs_blanc(i) = max(AreaBlanc);

    area_data_rouge{i} = AreaRouge;
    medians_rouge(i) = median(AreaRouge);
    q1_rouge(i) = quantile(AreaRouge, 0.25);
    q3_rouge(i) = quantile(AreaRouge, 0.75);
    mins_rouge(i) = min(AreaRouge);
    maxs_rouge(i) = max(AreaRouge);

    area_data_vert{i} = AreaVert;
    medians_vert(i) = median(AreaVert);
    q1_vert(i) = quantile(AreaVert, 0.25);
    q3_vert(i) = quantile(AreaVert, 0.75);
    mins_vert(i) = min(AreaVert);
    maxs_vert(i) = max(AreaVert);

    area_data_jaune{i} = AreaJaune;
    medians_jaune(i) = median(AreaJaune);
    q1_jaune(i) = quantile(AreaJaune, 0.25);
    q3_jaune(i) = quantile(AreaJaune, 0.75);
    mins_jaune(i) = min(AreaJaune);
    maxs_jaune(i) = max(AreaJaune);
end

% Affichage des résultats avec scatter et errorbar
figure; 
subplot(2,2,1); hold on;
x = 1:num_types;

% Points individuels
for i = 1:num_types
    scatter(ones(size(area_data_blanc{i})) * i, area_data_blanc{i}, 'b', 'filled', 'MarkerFaceAlpha', 0.3);
end

% Tracer les médianes
scatter(x, medians_blanc, 100, 'r', 'filled');

% Ajouter les barres d'erreur pour min-Q1-Q3-max
errorbar(x, medians_blanc, medians_blanc - q1_blanc, q3_blanc - medians_blanc, 'k', 'LineWidth', 2);

% Ajouter les min/max comme lignes verticales
for i = 1:num_types
    plot([i, i], [mins_blanc(i), maxs_blanc(i)], 'k--', 'LineWidth', 1.5);
end

xticks(x);
xticklabels(folders);
xlabel('Types de Pizza');
ylabel('Aire de la couleur (niveau de gris)');
title('Distribution du Blanc par Type de Pizza');
hold off;

subplot(2,2,2); hold on;
x = 1:num_types;

% Points individuels
for i = 1:num_types
    scatter(ones(size(area_data_rouge{i})) * i, area_data_rouge{i}, 'b', 'filled', 'MarkerFaceAlpha', 0.3);
end

% Tracer les médianes
scatter(x, medians_rouge, 100, 'r', 'filled');

% Ajouter les barres d'erreur pour min-Q1-Q3-max
errorbar(x, medians_rouge, medians_rouge - q1_rouge, q3_rouge - medians_rouge, 'k', 'LineWidth', 2);

% Ajouter les min/max comme lignes verticales
for i = 1:num_types
    plot([i, i], [mins_rouge(i), maxs_rouge(i)], 'k--', 'LineWidth', 1.5);
end

xticks(x);
xticklabels(folders);
xlabel('Types de Pizza');
ylabel('Aire de la couleur (niveau de gris)');
title('Distribution du Rouge par Type de Pizza');
hold off;

subplot(2,2,3); hold on;
x = 1:num_types;

% Points individuels
for i = 1:num_types
    scatter(ones(size(area_data_vert{i})) * i, area_data_vert{i}, 'b', 'filled', 'MarkerFaceAlpha', 0.3);
end

% Tracer les médianes
scatter(x, medians_vert, 100, 'r', 'filled');

% Ajouter les barres d'erreur pour min-Q1-Q3-max
errorbar(x, medians_vert, medians_vert - q1_vert, q3_vert - medians_vert, 'k', 'LineWidth', 2);

% Ajouter les min/max comme lignes verticales
for i = 1:num_types
    plot([i, i], [mins_vert(i), maxs_vert(i)], 'k--', 'LineWidth', 1.5);
end

xticks(x);
xticklabels(folders);
xlabel('Types de Pizza');
ylabel('Aire de la couleur (niveau de gris)');
title('Distribution du Vert par Type de Pizza');
hold off;

subplot(2,2,4); hold on;
x = 1:num_types;

% Points individuels
for i = 1:num_types
    scatter(ones(size(area_data_jaune{i})) * i, area_data_jaune{i}, 'b', 'filled', 'MarkerFaceAlpha', 0.3);
end

% Tracer les médianes
scatter(x, medians_jaune, 100, 'r', 'filled');

% Ajouter les barres d'erreur pour min-Q1-Q3-max
errorbar(x, medians_jaune, medians_jaune - q1_jaune, q3_jaune - medians_jaune, 'k', 'LineWidth', 2);

% Ajouter les min/max comme lignes verticales
for i = 1:num_types
    plot([i, i], [mins_jaune(i), maxs_jaune(i)], 'k--', 'LineWidth', 1.5);
end

xticks(x);
xticklabels(folders);
xlabel('Types de Pizza');
ylabel('Aire de la couleur (niveau de gris)');
title('Distribution du Jaune par Type de Pizza');
hold off;