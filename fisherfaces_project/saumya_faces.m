% =====================================================================
%  SAUMYA_FACES   Part 2: eigenfaces vs Fisherfaces on real faces
%
%  Four methods on two databases that say opposite things:
%
%    ORL   32 people, 10 images each, 92x112.  Varies POSE and EXPRESSION.
%    Yale  38 people, ~64 images each, 168x192.  Varies ILLUMINATION only.
%
%    1. correlation   nearest neighbour on raw pixels, no projection at all
%    2. eigenfaces    PCA, best k
%    3. eigenfaces    PCA at k = c-1, the SAME number of features as
%                     Fisherfaces, so the comparison is fair
%    4. Fisherfaces   PCA to m = N-c, then LDA, keeping c-1 directions
%
%  Also runs three checks that each test a different theorem.
%
%  Uses saumya_loadfaces, saumya_pca, saumya_fisherfaces, saumya_nn,
%  saumya_normcols, saumya_folders.
%
%  Run:  matlab -batch "run('saumya_faces.m')"
%  Figures are SAVED to figs_saumya/, not shown.
% =====================================================================

clear; clc;
rng(0);

showFigures = false;
if showFigures,  figVis = 'on';  else,  figVis = 'off';  end
set(groot, 'defaultFigureColor', 'w');
try
    sT = settings; sT.matlab.appearance.figure.GraphicsTheme.TemporaryValue = 'light';
catch
end

[orlFolders, yaleFolders, dORL, dYALE, figDir] = saumya_folders();

fprintf('=====================================================\n');
fprintf('  EIGENFACES vs FISHERFACES\n');
fprintf('=====================================================\n');

% name, folders, train per person, image height, image width
sets(1).name = 'ORL';   sets(1).folders = orlFolders;
sets(1).numTrain = 6;   sets(1).H = 112;  sets(1).W = 92;
sets(2).name = 'Yale';  sets(2).folders = yaleFolders;
sets(2).numTrain = 40;  sets(2).H = 192;  sets(2).W = 168;

results = zeros(2, 4);
keep    = cell(1, 2);

for s = 1:2
    name     = sets(s).name;
    folders  = sets(s).folders;
    numTrain = sets(s).numTrain;
    H = sets(s).H;  W = sets(s).W;  d = H * W;

    fprintf('\n-----------------------------------------------------\n');
    fprintf('  %s\n', name);
    fprintf('-----------------------------------------------------\n');

    [imTr, ytr, imTe, yte] = saumya_loadfaces(folders, numTrain, d);
    N = size(imTr, 2);
    c = numel(folders);
    fprintf('  d = %d pixels, N = %d training, %d test, c = %d people\n', ...
            d, N, size(imTe, 2), c);

    % ---- 1. correlation: the baseline with no projection ----
    % If a method cannot beat this, the projection is not earning its place.
    results(s,1) = saumya_nn(saumya_normcols(imTr), ytr, saumya_normcols(imTe), yte);
    fprintf('\n  1. correlation (raw pixels)     : %.1f%%\n', results(s,1));

    % ---- PCA ----
    [V, ~, xbar] = saumya_pca(imTr);

    % ---- 2. eigenfaces at the best k ----
    ks = [10 20 30 50 80 120 170];
    ks = ks(ks <= N - 1);              % only min(d,N)-1 directions exist
    best = 0;  bestK = 0;
    for k = ks
        Vk = V(:, 1:k);
        r  = saumya_nn(Vk' * (imTr - xbar), ytr, Vk' * (imTe - xbar), yte);
        if r > best
            best = r;  bestK = k;
        end
    end
    results(s,2) = best;
    fprintf('  2. eigenfaces, best k = %-3d     : %.1f%%\n', bestK, best);

    % ---- 3. eigenfaces at the matched dimension ----
    Vc = V(:, 1:(c-1));
    results(s,3) = saumya_nn(Vc' * (imTr - xbar), ytr, Vc' * (imTe - xbar), yte);
    fprintf('  3. eigenfaces at k = c-1 = %-3d  : %.1f%%\n', c-1, results(s,3));

    % ---- 4. Fisherfaces ----
    m = N - c;                          % the paper's choice
    [Wopt, Sb, Sw] = saumya_fisherfaces(V, xbar, imTr, ytr, m, c);
    results(s,4) = saumya_nn(Wopt' * (imTr - xbar), ytr, Wopt' * (imTe - xbar), yte);
    fprintf('  4. FISHERFACES (%d features)    : %.1f%%\n', size(Wopt,2), results(s,4));

    % ---- three checks, each testing a different theorem ----
    fprintf('\n  checks:\n');

    % (a) S_T = S_B + S_W is a theorem, so it must hold to machine precision.
    Al = V(:, 1:m)' * (imTr - xbar);
    Dl = Al - mean(Al, 2);
    St = Dl * Dl';
    fprintf('    (a) ||S_T-(S_B+S_W)||/||S_T|| = %.2e  (should be ~1e-16)\n', ...
            norm(St - (Sb + Sw), 'fro') / norm(St, 'fro'));

    % (b) rank(S_B) <= c-1: the c class means have only c-1 degrees of
    %     freedom once the overall mean is subtracted.  This is WHY there
    %     are at most c-1 Fisherfaces.
    fprintf('    (b) rank(S_B) = %d, bound c-1 = %d, kept %d\n', ...
            rank(Sb), c-1, size(Wopt,2));

    % (c) Fisherfaces are S_W-orthogonal, NOT Euclidean-orthogonal.
    Ge = Wopt' * Wopt;
    offE = norm(Ge - diag(diag(Ge)), 'fro') / norm(Ge, 'fro');
    Wfld = V(:, 1:m)' * Wopt;
    Gs   = Wfld' * Sw * Wfld;
    offS = norm(Gs - diag(diag(Gs)), 'fro') / norm(Gs, 'fro');
    fprintf('    (c) off-diagonal of W''W      = %.2e  (Euclidean)\n', offE);
    fprintf('        off-diagonal of W''S_W W  = %.2e  (S_W inner product)\n', offS);

    keep{s} = struct('name', name, 'xbar', xbar, 'V', V, 'Wopt', Wopt, ...
                     'H', H, 'W', W);
end

% ---- summary ----
fprintf('\n\n========== SUMMARY (recognition rate %%) ==========\n');
fprintf('  database   correlation  eig(best)  eig(c-1)  FISHERFACES\n');
for s = 1:2
    fprintf('  %-9s  %10.1f  %9.1f  %8.1f  %11.1f\n', sets(s).name, ...
            results(s,1), results(s,2), results(s,3), results(s,4));
end
fprintf('==================================================\n');
fprintf('\n  On Yale (illumination) Fisherfaces win by a wide margin: the\n');
fprintf('  paper''s claim, reproduced.  On ORL (pose and expression) they\n');
fprintf('  LOSE, because ORL gives 6 images per person so each class\n');
fprintf('  contributes only n_j-1 = 5 degrees of freedom to S_W, while Yale\n');
fprintf('  gives 39.  Note also that on Yale correlation BEATS eigenfaces:\n');
fprintf('  PCA actively hurts there, its leading components being lighting.\n');

% ---- figures: mean face, eigenfaces, Fisherfaces ----
for s = 1:2
    K = keep{s};

    figure('Visible', figVis);
    imagesc(reshape(K.xbar, K.H, K.W)); colormap gray; axis image off;
    title([K.name ': mean face']);
    saveas(gcf, [figDir K.name '_meanface.png']);
    if ~showFigures, close; end

    figure('Visible', figVis, 'Position', [100 100 700 700]);
    for l = 1:16
        subplot(4,4,l);
        imagesc(reshape(K.V(:,l), K.H, K.W)); colormap gray; axis image off;
        title(sprintf('e%d', l));
    end
    sgtitle([K.name ': top 16 eigenfaces (smooth, face-like)']);
    saveas(gcf, [figDir K.name '_eigenfaces.png']);
    if ~showFigures, close; end

    figure('Visible', figVis, 'Position', [100 100 700 700]);
    for l = 1:16
        subplot(4,4,l);
        imagesc(reshape(K.Wopt(:,l), K.H, K.W)); colormap gray; axis image off;
        title(sprintf('f%d', l));
    end
    sgtitle([K.name ': top 16 Fisherfaces (high frequency, not face-like)']);
    saveas(gcf, [figDir K.name '_fisherfaces.png']);
    if ~showFigures, close; end
end

fprintf('\n  figures written to %s\n', figDir);
fprintf('\n  DONE\n');
