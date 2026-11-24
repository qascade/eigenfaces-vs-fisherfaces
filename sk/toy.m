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
A = diag([4 0.5]);
X1 = A*randn(2,n) + [0; 1];      
X2 = A*randn(2,n) + [0; -1];

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

% 6. Classify by EUCLIDEAN DISTANCE to the projected class means.
%    Each point is one number after projection, so we ask which class mean it
%    is nearer to.  This is the same rule the face scripts use, and unlike a
%    threshold it assumes nothing about which class lands on which side.
accLDA = nearestMeanAccuracy(projLDA1, projLDA2);
fprintf('LDA direction wLDA = (%.3f, %.3f), accuracy = %.1f%%\n', wLDA, accLDA);

% For comparison: PCA direction = top eigenvector of the covariance
X = [X1, X2];
[V, D] = eig(cov(X'));
[~, i] = max(diag(D));                    % eig is ASCENDING, so take the max
wPCA = V(:, i);

projPCA1 = wPCA' * X1;
projPCA2 = wPCA' * X2;
accPCA = nearestMeanAccuracy(projPCA1, projPCA2);
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
histogram(projPCA1, 30);  histogram(projPCA2, 30);
title(sprintf('Projected onto PCA: overlap (%.1f%%)', accPCA)); grid on;

subplot(1, 3, 3); hold on;
histogram(projLDA1, 30);  histogram(projLDA2, 30);
title(sprintf('Projected onto LDA: separated (%.1f%%)', accLDA)); grid on;

saveas(gcf, [figDir 'toy.png']);
fprintf('figure saved to %s\n', figDir);


% ---------------------------------------------------------------------
function acc = nearestMeanAccuracy(proj1, proj2)
% Assign each projected point to the nearer class mean, by squared Euclidean
% distance.  We compare SQUARED distances and never take a square root,
% because sqrt is increasing and so cannot change which mean is nearer --
% the same shortcut the face scripts use.
%
% In one dimension this is exactly the midpoint-threshold rule, so the
% accuracy is unchanged; written this way it generalises to any number of
% dimensions and needs no sign convention.

centre1 = mean(proj1);
centre2 = mean(proj2);

correct = 0;
for i = 1:numel(proj1)
    if (proj1(i) - centre1)^2 <= (proj1(i) - centre2)^2
        correct = correct + 1;          % class 1 point, nearer to centre1
    end
end
for i = 1:numel(proj2)
    if (proj2(i) - centre2)^2 <= (proj2(i) - centre1)^2
        correct = correct + 1;          % class 2 point, nearer to centre2
    end
end

acc = 100 * correct / (numel(proj1) + numel(proj2));
end
