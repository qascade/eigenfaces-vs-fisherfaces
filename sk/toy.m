%% Minimal LDA on toy data (two classes in 2D)
% Shows why PCA is not a classifier: all the SPREAD is along x, but all the
% CLASS INFORMATION is along y.  PCA maximises spread and picks x.  Fisher
% maximises (between-class / within-class) and picks y.
%
% Notation follows the study guide: S_W within-class scatter, S_B between,
% w the projection direction, J(w) the Fisher criterion.

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
m1 = mean(X1, 2);
m2 = mean(X2, 2);

% 3. Within-class scatter: spread of each class around its own mean
Sw = (X1 - m1)*(X1 - m1)' + (X2 - m2)*(X2 - m2)';

% 4. LDA direction: w = Sw^(-1) (m1 - m2), made unit length.
%    Sw \ (m1-m2) SOLVES Sw*w = m1-m2; it never forms an inverse.
%    J(w) is unchanged if w is scaled, so the length is ours to choose.
w = Sw \ (m1 - m2);
w = w / norm(w);

% 5. Project every point onto w (each point becomes one number)
y1 = w' * X1;
y2 = w' * X2;

% 6. Classify with a threshold halfway between the projected means
t = (mean(y1) + mean(y2)) / 2;
acc = 100 * (sum(y1 > t) + sum(y2 <= t)) / (2*n);
fprintf('LDA direction w = (%.3f, %.3f), accuracy = %.1f%%\n', w, acc);

% For comparison: PCA direction = top eigenvector of the covariance
X = [X1, X2];
[V, D] = eig(cov(X'));
[~, i] = max(diag(D));                    % eig is ASCENDING, so take the max
wp = V(:, i);
accp = 100 * (sum(wp'*X1 > (mean(wp'*X1)+mean(wp'*X2))/2) + ...
              sum(wp'*X2 <= (mean(wp'*X1)+mean(wp'*X2))/2)) / (2*n);
fprintf('PCA direction   = (%.3f, %.3f), accuracy = %.1f%%\n', wp, accp);

% 7. Plots
figure('Position', [100 100 1100 350]);

subplot(1, 3, 1); hold on;
plot(X1(1,:), X1(2,:), 'b.');
plot(X2(1,:), X2(2,:), 'r.');
quiver(0, 0, 6*wp(1), 6*wp(2), 0, 'm', 'LineWidth', 2);
quiver(0, 0, 3*w(1),  3*w(2),  0, 'g', 'LineWidth', 2);
legend('class 1', 'class 2', 'PCA direction', 'LDA direction');
axis equal; grid on; title('Data and directions');

subplot(1, 3, 2); hold on;
histogram(wp' * X1, 30);  histogram(wp' * X2, 30);
title(sprintf('Projected onto PCA: overlap (%.1f%%)', accp)); grid on;

subplot(1, 3, 3); hold on;
histogram(y1, 30);  histogram(y2, 30);
title(sprintf('Projected onto LDA: separated (%.1f%%)', acc)); grid on;

saveas(gcf, [figDir 'toy.png']);
fprintf('figure saved to %s\n', figDir);
