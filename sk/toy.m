%% Minimal LDA on toy data (two classes in 2D)
% Shows why PCA is not a classifier: all the SPREAD is along x, but all the
% CLASS INFORMATION is along y.  PCA maximises spread and picks x.  Fisher
% maximises (between-class / within-class) and picks y.
%
% Notation follows the study guide: S_W within-class scatter, S_B between,
% wLDA the projection direction, J(wLDA) the Fisher criterion.

clc; clear; close all;
rng(1);                                   % same data every run
n = 200;                                  % points per class

figDir = 'figures/';
if exist(figDir, 'dir') == 0
    mkdir(figDir);
end

% 1. Make data: both classes are wide in x, thin in y,
%    and they differ only in y (class 1 at y = +1, class 2 at y = -1)
X1 = [4*randn(1, n); 0.5*randn(1, n) + 1];
X2 = [4*randn(1, n); 0.5*randn(1, n) - 1];

% 2. Class means
mean1 = mean(X1, 2);
mean2 = mean(X2, 2);

% 3. Within-class scatter: spread of each class around its own mean
Sw = (X1 - mean1)*(X1 - mean1)' + (X2 - mean2)*(X2 - mean2)';

% 4. LDA direction: wLDA = Sw^(-1) (mean1 - mean2), made unit length.
%    Sw \ (mean1-mean2) SOLVES Sw*wLDA = mean1-mean2; it never forms an inverse.
%    J(wLDA) is unchanged if wLDA is scaled, so the length is ours to choose.
wLDA = Sw \ (mean1 - mean2);
wLDA = wLDA / norm(wLDA);

% 5. Project every point onto wLDA (each point becomes one number)
projLDA1 = wLDA' * X1;
projLDA2 = wLDA' * X2;

% 6. Classify with a threshold halfway between the projected means
threshold = (mean(projLDA1) + mean(projLDA2)) / 2;
accLDA = 100 * (sum(projLDA1 > threshold) + sum(projLDA2 <= threshold)) / (2*n);
fprintf('LDA direction wLDA = (%.3f, %.3f), accuracy = %.1f%%\n', wLDA, accLDA);

% For comparison: PCA direction = top eigenvector of the covariance
X = [X1, X2];
[V, D] = eig(cov(X'));
[~, i] = max(diag(D));                    % eig is ASCENDING, so take the max
wPCA = V(:, i);
accPCA = 100 * (sum(wPCA'*X1 > (mean(wPCA'*X1)+mean(wPCA'*X2))/2) + ...
              sum(wPCA'*X2 <= (mean(wPCA'*X1)+mean(wPCA'*X2))/2)) / (2*n);
fprintf('PCA direction   = (%.3f, %.3f), accuracy = %.1f%%\n', wPCA, accPCA);

% 7. Plots
figure('Position', [100 100 1100 350]);

subplot(1, 3, 1); hold on;
plot(X1(1,:), X1(2,:), 'b.');
plot(X2(1,:), X2(2,:), 'r.');
quiver(0, 0, 6*wPCA(1), 6*wPCA(2), 0, 'm', 'LineWidth', 2);
quiver(0, 0, 3*wLDA(1),  3*wLDA(2),  0, 'g', 'LineWidth', 2);
legend('class 1', 'class 2', 'PCA direction', 'LDA direction');
axis equal; grid on; title('Data and directions');

subplot(1, 3, 2); hold on;
histogram(wPCA' * X1, 30);  histogram(wPCA' * X2, 30);
title(sprintf('Projected onto PCA: overlap (%.1f%%)', accPCA)); grid on;

subplot(1, 3, 3); hold on;
histogram(projLDA1, 30);  histogram(projLDA2, 30);
title(sprintf('Projected onto LDA: separated (%.1f%%)', accLDA)); grid on;

saveas(gcf, [figDir 'toy.png']);
fprintf('figure saved to %s\n', figDir);
