function mpg_regression()
%MPG_REGRESSION MPG regression assignment
%   Code submission by: Z0966990

% Name indices of data.
VOL = 1;
HP = 2;
MPG = 3;
SP = 4;
WT = 5;
GPM = 6;

%% Specify regression models.
models = {
    MPG, [VOL, HP, SP, WT];
    MPG, [HP, WT];
    MPG, [SP, WT];
    MPG, HP;
    MPG, SP;
    MPG, WT;
    GPM, [VOL, HP, SP, WT];
    GPM, [HP, WT];
    GPM, [SP, WT];
    GPM, HP;
    GPM, SP;
    GPM, WT;
    };

%% Load data from this directory
data = importdata('carmpgdat.txt', '\t', 1);
names = data.textdata(1, 2:end);
data = data.data;

data(:,GPM) = 1./data(:,MPG);
names{GPM} = 'GPM';

%% Scatter plots.
figure();

ax = subplot(2, 2, 1);
scatter(ax, data(:,MPG), data(:,HP), '+');
title(ax, 'HP vs MPG');
xlabel(ax, 'Miles Per Gallon (MPG)');
ylabel(ax, 'Horse Power (HP)');

ax = subplot(2, 2, 2);
scatter(ax, data(:,MPG), data(:,WT), '+');
title(ax, 'WT vs MPG');
xlabel(ax, 'Miles Per Gallon (MPG)');
ylabel(ax, 'Weight (WT)');

ax = subplot(2, 2, 3);
scatter(ax, data(:,GPM), data(:,HP), '+');
title(ax, 'HP vs GPM');
xlabel(ax, 'Gallons Per Mile (GPM)');
ylabel(ax, 'Horse Power (HP)');

ax = subplot(2, 2, 4);
scatter(ax, data(:,GPM), data(:,WT), '+');
title(ax, 'WT vs GPM');
xlabel(ax, 'Gallons Per Mile (GPM)');
ylabel(ax, 'Weight (WT)');


%% Regression modelling.
model_names = cell(size(models, 1), 1);
R2 = zeros(size(models, 1), 1);
R2_adj = zeros(size(models, 1), 1);
p = zeros(size(models, 1), 1);
ks_result = zeros(size(models, 1), 1);
ks_p_value = zeros(size(models, 1), 1);
e = zeros(size(data, 1), size(models, 1));
for i = 1:size(models, 1)
    % Parse model definition.
    [y_idx, X_idx] = models{i, :};
    y = data(:, y_idx);
    X = data(:, X_idx);
    [n, p(i)] = size(X);
    
    % Determine name.
    model_names{i} = sprintf('%s vs %s',...
        strjoin(names(X_idx), ', '), names{y_idx});
    
    % Calculate model.
    b = regress(y, X);
    b0 = mean(y) - mean(X)*b;
    
    % Calculate standardised residuals.
    yhat = b0 + X*b;
    e(:, i) = y - yhat;
    e_std = (e(:, i) - mean(e(:, i)))/std(e(:, i));
    
    % Test residuals.
    [ks_result(i), ks_p_value(i)] = kstest(e_std);
    
    % Find R2
    SS_e = sumsqr(e(:, i));
    SS_T = sumsqr(y);
    R2(i) = 1 - SS_e/SS_T;
    R2_adj(i) = 1 - SS_e*(n-1)/(SS_T*(n-p(i)));
end

%% Generate results table.
disp(table(R2, R2_adj, p, ks_result, ks_p_value, 'RowNames', model_names));

%% Plot residuals for best and worst models.
% Evaluate models using R2_adj.
[~, worst_idx] = min(R2_adj);
worst_idx = worst_idx(1);
[~, best_idx] = max(R2_adj);
best_idx = best_idx(1);

% Residual plots.
figure();

ax = subplot(1, 2, 1); 
histogram(e(:, worst_idx), 'Normalization', 'probability');
title(sprintf('Residuals for %s', model_names{worst_idx}));
xlabel('Residuals');

ax = subplot(1, 2, 2);
histogram(e(:, best_idx), 'Normalization', 'probability');
title(sprintf('Residuals for %s', model_names{best_idx}));
xlabel('Residuals');
end