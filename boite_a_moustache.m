clc; clear; close all;

%% Boite à moustache : Blanc, Rouge, Vert, Jaune, Marron et Rose

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
        Areas(j, :) = (caraccouleur(img))';
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

for c = 1:6
    subplot(3,2,c); hold on;
    x = 1:num_types;

    % Réinitialiser les données Excel pour chaque couleur
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

% Convertir en table avec des noms de colonnes uniques
excelTable = cell2table(excelData, 'VariableNames', {'RowIdx', 'ColumnIdx', 'ElementIdx', 'Value'});

% Générer un fichier unique par couleur
filename = sprintf('output_color.xlsx');
writetable(excelTable, filename);

%% Classification sur la couleur & Matrice de confusion correspondante

while (accuracy<0.7)

    folders = {'pizzafromag', 'pizzahawai', 'pizzamargherita', 'pizzapepperoni', 'pizzareine', 'pizzavege'};
    num_types = numel(folders);
    
    % Initialisation des caractéristiques et des labels
    features = [];
    labels = [];
    
    % Parcours des dossiers
    for i = 1:num_types
        files = dir(fullfile('masked_dataset\', folders{i}, '*.jpg'));
        num_files = numel(files);
    
        for j = 1:num_files
            img = imread(fullfile('masked_dataset\', folders{i}, files(j).name));
            img_size = size(img,1) * size(img,2);
    
            % Extraire les caractéristiques (proportions de couleur)
            feat_vec = caraccouleur(img);
    
            % Stocker les caractéristiques et les labels
            features = [features; feat_vec'];
            labels = [labels; i]; % Label numérique correspondant au type de pizza
        end
    end
    
    cv = cvpartition(size(features, 1), 'HoldOut', 0.2);
    train_idx = training(cv);
    test_idx = test(cv);
    
    X_train = features(train_idx, :);
    y_train = labels(train_idx);
    X_test = features(test_idx, :);
    y_test = labels(test_idx);
    
    % Création du modèle k-NN
    knn_model = fitcknn(X_train, y_train, 'NumNeighbors', 6);
    
    % Prédiction sur l'ensemble de test
    y_pred = predict(knn_model, X_test);
    
    % Matrice de confusion et précision
    confusion_matrix = confusionmat(y_test, y_pred);
    accuracy = sum(diag(confusion_matrix)) / sum(confusion_matrix(:));
    
    % Affichage des résultats
    disp('Matrice de confusion :');
    disp(confusion_matrix);
    fprintf('Précision globale : %.2f%%\n', accuracy * 100);
    
end