clc; clear; close all;

%% Boite à moustache : Blanc, Rouge, Vert, Jaune et Marron

% Définition des dossiers contenant les images
folders = {'pizzafromag', 'pizzahawai', 'pizzamargherita', 'pizzapepperoni', 'pizzareine', 'pizzavege'};
num_types = numel(folders);

area_data = cell(num_types, 6);
medians = zeros(num_types, 6);
q1 = zeros(num_types, 6);
q3 = zeros(num_types, 6);
mins = zeros(num_types, 6);
maxs = zeros(num_types, 6);

% Parcours des dossiers
for i = 1:num_types
    files = dir(fullfile('masked_dataset\', folders{i}, '*.jpg')); 
    num_files = numel(files);
    Areas = zeros(num_files, 6);
    
    for j = 1:num_files
        img = imread(fullfile('masked_dataset\', folders{i}, files(j).name));
        Areas(j, 1) = caracblanc(img) / (size(img,1) * size(img,2));
        Areas(j, 2) = caracrouge(img) / (size(img,1) * size(img,2));
        Areas(j, 3) = caracvert(img) / (size(img,1) * size(img,2));
        Areas(j, 4) = caracjaune(img) / (size(img,1) * size(img,2));
        Areas(j, 5) = caracmarron(img) / (size(img,1) * size(img,2));
        Areas(j, 6) = caracrose(img) / (size(img,1) * size(img,2));
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
colors = {'Blanc', 'Rouge', 'Vert', 'Jaune', 'Marron', 'Rose'};

for c = 1:5
    subplot(3,2,c); hold on;
    x = 1:num_types;

    % Créer un cell array pour stocker les données à écrire dans Excel
    excelData = {};
    
    rowIndex = 1; % Indice de ligne pour l'Excel
    
    for i = 1:size(area_data, 1)
        for j = 1:size(area_data, 2)
            values = area_data{i, j}; % Extraire le tableau de la cellule
            
            for k = 1:numel(values)
                % Ajouter les données sous forme de lignes dans excelData
                excelData{rowIndex, 1} = i;  % Indice ligne cellule
                excelData{rowIndex, 2} = j;  % Indice colonne cellule
                excelData{rowIndex, 3} = k;  % Indice élément dans le tableau
                excelData{rowIndex, 4} = values(k); % Valeur
                
                rowIndex = rowIndex + 1; % Passer à la ligne suivante
            end
        end
    end
    
    % Convertir en table pour un meilleur affichage dans Excel
    excelTable = cell2table(excelData, 'VariableNames', {'Row', 'Column', 'Index', 'Value'});
    
    % Écrire dans un fichier Excel
    writetable(excelTable, 'output.xlsx');
    
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
