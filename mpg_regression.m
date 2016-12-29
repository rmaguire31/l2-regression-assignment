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
plot_variables(subplot(1, 2, 1), HP, MPG);
plot_variables(subplot(1, 2, 2), WT, MPG);
figure();
plot_variables(subplot(1, 2, 1), HP, GPM);
plot_variables(subplot(1, 2, 2), WT, GPM);

%% Regression modelling.
R2 = zeros(size(models, 1), 1);
R2_adj = zeros(size(models, 1), 1);
p = zeros(size(models, 1), 1);
ks_test = zeros(size(models, 1), 1);
ks_p_value = zeros(size(models, 1), 1);
e = zeros(size(data, 1), size(models, 1));
y_name = cell(size(models, 1), 1);
X_names = cell(size(models, 1), 1);
for i = 1:size(models, 1)
    % Parse model definition.
    [y_idx, X_idx] = models{i, :};
    y = data(:, y_idx);
    X = data(:, X_idx);
    [n, p(i)] = size(X);
    
    % Determine name.
    y_name{i}= names{y_idx};
    X_names{i} = strjoin(names(X_idx), ', ');
    
    % Calculate model.
    b = regress(y, X);
    a = mean(y) - mean(X)*b;
    
    % Calculate standardised residuals.
    yhat = a + X*b;
    e(:, i) = y - yhat;
    e_std = (e(:, i) - mean(e(:, i)))/std(e(:, i));
    
    % Test residuals.
    [ks_test(i), ks_p_value(i)] = kstest(e_std);
    
    % Find R2
    SS_e = sumsqr(e(:, i));
    SS_T = sumsqr(y);
    R2(i) = 1 - SS_e/SS_T;
    R2_adj(i) = 1 - SS_e*(n-1)/(SS_T*(n-p(i)));
end

%% Generate results table.
disp(table(y_name, X_names, R2, R2_adj, p, ks_test, ks_p_value));

%% Plot residuals for best and worst models.
% Evaluate models using R2_adj and rename best and worst.
[~, worst_idx] = min(R2_adj);
worst_idx = worst_idx(1);

[~, best_idx] = max(R2_adj);
best_idx = best_idx(1);

% Plot residuals for best and worst models.
figure();
plot_residuals(subplot(1, 2, 1), worst_idx, 'Worst Regression');
plot_residuals(subplot(1, 2, 2), best_idx, 'Best Regression');

    function plot_variables(ax, X_idx, y_idx)
    %%
    % Make title bold.
    titletxt = sprintf('%s vs %s', names{X_idx}, names{y_idx});
    scatter(ax, data(:,X_idx), data(:,y_idx), '+');
    title(ax, titletxt, 'Interpreter', 'latex', 'FontSize', 36);
    xlabel(ax, names{X_idx}, 'Interpreter', 'latex', 'FontSize', 34);
    ylabel(ax, names{y_idx}, 'Interpreter', 'latex', 'FontSize', 34);
    ax.FontSize = 30;
    ax.TickLabelInterpreter = 'latex';
    end

    function plot_residuals(ax, idx, titletxt)
    %%
    % Add second line to title.
    titletxt = {
        sprintf('\\makebox[4in][c]{%s}', titletxt)
        sprintf('\\makebox[4in][c]{%s(%s)}', y_name{idx}, X_names{idx})};
    histogram(ax, e(:, idx), 'Normalization', 'probability');
    title(ax, titletxt, 'Interpreter', 'latex', 'FontSize', 36);
    xlabel(ax, sprintf('Residual %s', y_name{idx}),...
        'Interpreter', 'latex', 'FontSize', 34);
    ylabel(ax, 'Density', 'Interpreter', 'latex', 'FontSize', 34);
    ax.FontSize = 30;
    ax.TickLabelInterpreter = 'latex';
    end
end