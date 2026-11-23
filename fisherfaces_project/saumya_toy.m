% =====================================================================
%  SAUMYA_TOY   Part 1: why PCA is not a classifier, in 2-D
%
%  Two classes of 2-D points sharing a covariance
%        Sigma = [16 0; 0 0.25]
%  so each cloud is a wide flat ellipse stretched along x.  The means differ
%  only in y:  mu1 = (0,-1), mu2 = (0,+1).
%
%  So ALL the spread is along x and ALL the class information is along y.
%  PCA maximises spread, so it picks x and destroys the labels.
%  Fisher maximises (between / within), so it picks y.
%
%  That is the paper's whole argument, in two dimensions.
%
%  Run:  matlab -batch "run('saumya_toy.m')"
%  The figure is SAVED to figs_saumya/, not shown.  Set showFigures = true
%  and run from the MATLAB desktop to see it on screen.
% =====================================================================

clear; clc;
rng(0);                        % fixed seed, so the numbers are repeatable

showFigures = false;
if showFigures,  figVis = 'on';  else,  figVis = 'off';  end
set(groot, 'defaultFigureColor', 'w');
try
    sT = settings; sT.matlab.appearance.figure.GraphicsTheme.TemporaryValue = 'light';
catch
end

[~, ~, ~, ~, figDir] = saumya_folders();

fprintf('=====================================================\n');
fprintf('  TOY PROBLEM: PCA vs Fisher in 2-D\n');
fprintf('=====================================================\n');

Sigma = [16 0; 0 0.25];
mu1   = [ 0; -1];
mu2   = [ 0;  1];
nPer  = 500;

% Generate the clouds.  If z has covariance I then A*z has covariance A*A'.
% Cholesky gives Sigma = R'*R, so taking A = R' gives A*A' = R'*R = Sigma.
R  = chol(Sigma);
X1 = mu1 + R' * randn(2, nPer);
X2 = mu2 + R' * randn(2, nPer);

% ---- the two scatter matrices, straight from the definitions ----
m1   = mean(X1, 2);
m2   = mean(X2, 2);
mAll = (m1 + m2) / 2;

D1 = X1 - m1;                  % each class centred on ITS OWN mean
D2 = X2 - m2;
Sw = D1 * D1' + D2 * D2';      % within-class scatter

Sb = nPer * (m1 - mAll) * (m1 - mAll)' + ...
     nPer * (m2 - mAll) * (m2 - mAll)';      % between-class scatter

% ---- Fisher's direction: the two-class closed form ----
% Maximising J(w) = (w'*Sb*w)/(w'*Sw*w) for TWO classes gives
%        w  proportional to  inv(Sw)*(m1 - m2)
% Sw\(m1-m2) SOLVES Sw*w = (m1-m2); it does not form an inverse.
wLDA = Sw \ (m1 - m2);
wLDA = wLDA / norm(wLDA);      % J is scale invariant, so length is free

% ---- PCA's direction: top eigenvector of the TOTAL scatter ----
Xall = [X1, X2];
Dall = Xall - mean(Xall, 2);
St   = Dall * Dall';
[Vt, Dt]  = eig(St);
[~, iBig] = max(diag(Dt));     % eig is ASCENDING, so take the max
wPCA = Vt(:, iBig);

% ---- score both directions ----
Jlda = (wLDA' * Sb * wLDA) / (wLDA' * Sw * wLDA);
Jpca = (wPCA' * Sb * wPCA) / (wPCA' * Sw * wPCA);

fprintf('\n  Fisher direction : (%.4f, %.4f)   <- the y-axis\n', wLDA);
fprintf('  PCA    direction : (%.4f, %.4f)   <- the x-axis\n', wPCA);
fprintf('  J(w_Fisher) = %.4f\n', Jlda);
fprintf('  J(w_PCA)    = %.4f\n', Jpca);
fprintf('  ratio       = %.0fx\n', Jlda / Jpca);

% ---- turn J into something interpretable ----
% Project to 1-D, classify by the nearer class mean, count mistakes.
[errLDA, dpLDA] = toyScore(X1, X2, wLDA);
[errPCA, dpPCA] = toyScore(X1, X2, wPCA);

% d' is the gap between the projected means measured in within-class standard
% deviations.  For two classes J and d' are the same statement:
%       d' = 2*sqrt(J)      and      error = Phi(-sqrt(J))
% Phi(-t) = 0.5*erfc(t/sqrt(2)); MATLAB has no normcdf without the toolbox.
predLDA = 100 * 0.5 * erfc(sqrt(Jlda) / sqrt(2));
predPCA = 100 * 0.5 * erfc(sqrt(Jpca) / sqrt(2));

fprintf('\n  direction   d''      measured error   predicted Phi(-sqrt(J))\n');
fprintf('  Fisher    %5.2f      %6.2f%%           %6.2f%%\n', dpLDA, errLDA, predLDA);
fprintf('  PCA       %5.2f      %6.2f%%           %6.2f%%\n', dpPCA, errPCA, predPCA);
fprintf('  (2*sqrt(J) = %.2f for Fisher, matching d'')\n', 2*sqrt(Jlda));

% ---- the floor nothing can beat ----
% For equal-covariance Gaussians with equal priors the best possible error is
% Phi(-Delta/2), where Delta is the Mahalanobis distance between the means.
Delta = sqrt((mu2 - mu1)' * (Sigma \ (mu2 - mu1)));
bayes = 100 * 0.5 * erfc((Delta / 2) / sqrt(2));
fprintf('\n  Mahalanobis separation Delta = %.2f\n', Delta);
fprintf('  Bayes error (the floor)      = %.2f%%\n', bayes);
fprintf('  Measuring slightly BELOW the floor is sampling noise on %d\n', 2*nPer);
fprintf('  points, not a contradiction: nothing can beat the Bayes error.\n');

% ---- figure ----
figure('Visible', figVis, 'Position', [100 100 1000 380]);

subplot(1, 3, 1);
plot(X1(1,:), X1(2,:), '.', 'MarkerSize', 4); hold on;
plot(X2(1,:), X2(2,:), '.', 'MarkerSize', 4);
t = linspace(-12, 12, 2);
plot(t*wLDA(1), t*wLDA(2), 'k-',  'LineWidth', 2);
plot(t*wPCA(1), t*wPCA(2), 'k--', 'LineWidth', 2);
hold off; axis equal; grid on;
xlabel('x'); ylabel('y');
legend({'class 1','class 2','Fisher','PCA'}, 'Location','southoutside');
title('the two clouds and the two directions');

subplot(1, 3, 2);
histogram(wLDA' * X1, 40); hold on; histogram(wLDA' * X2, 40); hold off;
grid on; xlabel('projection onto Fisher direction');
title(sprintf('Fisher: %.2f%% error', errLDA));

subplot(1, 3, 3);
histogram(wPCA' * X1, 40); hold on; histogram(wPCA' * X2, 40); hold off;
grid on; xlabel('projection onto PCA direction');
title(sprintf('PCA: %.2f%% error', errPCA));

saveas(gcf, [figDir 'toy.png']);
if ~showFigures
    close;
end
fprintf('\n  figure written to %s\n', figDir);


% ---------------------------------------------------------------------
function [err, dprime] = toyScore(X1, X2, w)
% Project both clouds onto w, classify by the nearer projected class mean,
% and also report d' = |mean gap| / pooled within-class standard deviation.

p1 = w' * X1;                  % 1 x n row of projections
p2 = w' * X2;
c1 = mean(p1);
c2 = mean(p2);

wrong = 0;
for i = 1:numel(p1)
    if abs(p1(i) - c1) > abs(p1(i) - c2)      % closer to the other mean
        wrong = wrong + 1;
    end
end
for i = 1:numel(p2)
    if abs(p2(i) - c2) > abs(p2(i) - c1)
        wrong = wrong + 1;
    end
end
err = 100 * wrong / (numel(p1) + numel(p2));

n1 = numel(p1);
n2 = numel(p2);
sp = sqrt(((n1-1)*var(p1) + (n2-1)*var(p2)) / (n1 + n2 - 2));
dprime = abs(c1 - c2) / sp;
end
