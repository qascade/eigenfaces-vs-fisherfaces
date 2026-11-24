clc; clear; close all;
rng(1);                                   
n = 200;                                  


figDir = 'figures/';
if exist(figDir, 'dir') == 0
    mkdir(figDir);
end

A = diag([4 0.5]);
X1 = A*randn(2,n) + [0; 1];      
X2 = A*randn(2,n) + [0; -1];

% 1. Class means
mean1 = mean(X1, 2);
mean2 = mean(X2, 2);

% 2. Within-class scatter: spread of each class around its own mean
Sw = (X1 - mean1)*(X1 - mean1)' + (X2 - mean2)*(X2 - mean2)';

% 3. LDA direction: wLDA = Sw^(-1) (mean1 - mean2)
wLDA = Sw \ (mean1 - mean2);
wLDA = wLDA / norm(wLDA);

% 4. Project every point onto wLDA (each point becomes one number)
projLDA1 = wLDA' * X1;
projLDA2 = wLDA' * X2;


accLDA = nearestMeanAccuracy(projLDA1, projLDA2);
fprintf('LDA direction wLDA = (%.3f, %.3f), accuracy = %.1f%%\n', wLDA, accLDA);

% For comparison: PCA direction = top eigenvector of the covariance
X = [X1, X2];
C = cov(X');
[V, D] = eig(C);
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
