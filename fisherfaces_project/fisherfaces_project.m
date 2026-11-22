% =====================================================================
%  CS 740 PROJECT -- EIGENFACES vs FISHERFACES
%
%  Belhumeur, Hespanha & Kriegman, "Eigenfaces vs. Fisherfaces:
%  Recognition Using Class Specific Linear Projection", IEEE TPAMI
%  19(7), 1997.
%
%  Shubh Karman Singh (26M0843), Saumya Bajpai (26D0366)
%
%  This is ONE monolithic script.  Every experiment is a numbered
%  section (%% S0, %% S1, ...) so that any single part can be re-run on
%  its own.  All helper routines are LOCAL FUNCTIONS collected at the
%  very bottom of the file (MATLAB requires local functions in a script
%  to come last).
%
%  Run with:   matlab -batch "run('fisherfaces_project.m')"
% =====================================================================

clc;
clear;
close all;

%% ====================================================================
%% S0  CONFIGURATION
%  Paths are relative to this file, which lives in hw3/fisherfaces_project/.
%  rng(0) fixes the random number generator so that the "random split"
%  experiments give the same answer every time we run the script -- without
%  this, every run would print different recognition rates and we could not
%  quote stable numbers in the report.
%% ====================================================================

orlRoot  = '../ORL/';
yaleRoot = '../CroppedYale/';
figDir   = 'figs/';

if exist(figDir, 'dir') == 0
    mkdir(figDir);
end

rng(0);

% MATLAB R2025a and later choose a figure colour theme automatically, and on this
% machine that resolves to the DARK theme -- which produces black-background
% figures that look wrong printed in a report.  Force the light theme, and also
% force a white figure background for the image montages.  Wrapped in try/catch
% because the settings path does not exist on older MATLAB versions (where
% figures are light by default anyway).
try
    sTheme = settings;
    sTheme.matlab.appearance.figure.GraphicsTheme.TemporaryValue = 'light';
catch
    % older MATLAB: nothing to do
end
set(groot, 'defaultFigureColor', 'w');

% Image sizes (pixels).  d = number of pixels = length of one image vector.
orlH = 112;  orlW = 92;   dORL  = orlH * orlW;    % 10304
yalH = 192;  yalW = 168;  dYALE = yalH * yalW;    % 32256

fprintf('=====================================================\n');
fprintf('  EIGENFACES vs FISHERFACES\n');
fprintf('=====================================================\n');

%% ====================================================================
%% S1  TOY 2-D EXPERIMENT:  WHERE PCA FAILS AND LDA SUCCEEDS
%
%  Two Gaussian classes that share the same covariance matrix.
%      Sigma = [16 0; 0 0.25]   mu1 = (0,-1)   mu2 = (0,+1)
%
%  The spread WITHIN each class is huge along x (variance 16) and tiny
%  along y (variance 0.25).  The two classes are separated ONLY along y.
%
%  PCA maximises total variance, so it will pick x -- the direction that
%  carries no class information at all.
%  LDA maximises the ratio (between-class)/(within-class), so it picks y.
%
%  This is the whole Eigenfaces-vs-Fisherfaces story in two dimensions.
%% ====================================================================

fprintf('\n--- S1: toy 2-D PCA vs LDA ---\n');

Sigma = [16 0; 0 0.25];
mu1   = [0; -1];
mu2   = [0;  1];
nToy  = 500;

% To draw a sample from N(mu, Sigma) we need a matrix Rt with Rt*Rt' = Sigma.
% chol gives Sigma = R'*R, so Rt = R' works:  if z ~ N(0,I) then
% cov(R'z) = R'*I*R = R'*R = Sigma.
R = chol(Sigma);

X1 = zeros(2, nToy);
X2 = zeros(2, nToy);
for i = 1:nToy
    X1(:, i) = mu1 + R' * randn(2, 1);
    X2(:, i) = mu2 + R' * randn(2, 1);
end

Xtoy = [X1, X2];
ytoy = [ones(1, nToy), 2 * ones(1, nToy)];

% ---- sample means ----
m1 = mean(X1, 2);
m2 = mean(X2, 2);
mAll = mean(Xtoy, 2);

% ---- within-class scatter S_W = sum over classes of sum (x-mi)(x-mi)' ----
D1 = X1 - m1;
D2 = X2 - m2;
SwToy = D1 * D1' + D2 * D2';

% ---- between-class scatter S_B = sum over classes of ni (mi-m)(mi-m)' ----
SbToy = nToy * (m1 - mAll) * (m1 - mAll)' + nToy * (m2 - mAll) * (m2 - mAll)';

% ---- LDA direction: w proportional to inv(S_W) * (m2 - m1) ----
% The backslash solves SwToy * w = (m2-m1) without forming the inverse,
% which is both faster and numerically better than inv(SwToy)*(m2-m1).
wLDA = SwToy \ (m2 - m1);
wLDA = wLDA / norm(wLDA);

% ---- PCA direction: top eigenvector of the TOTAL covariance ----
Xtoy_c  = Xtoy - mAll;
Ctoy  = (Xtoy_c * Xtoy_c') / size(Xtoy, 2);
[Vt, Dt] = eig(Ctoy);
[~, it]  = sort(diag(Dt), 'descend');
wPCA = Vt(:, it(1));
wPCA = wPCA / norm(wPCA);

fprintf('  LDA direction : (%.4f, %.4f)\n', wLDA(1), wLDA(2));
fprintf('  PCA direction : (%.4f, %.4f)\n', wPCA(1), wPCA(2));

% ---- Fisher criterion J(w) = (w'*Sb*w) / (w'*Sw*w) ----
jLDA = (wLDA' * SbToy * wLDA) / (wLDA' * SwToy * wLDA);
jPCA = (wPCA' * SbToy * wPCA) / (wPCA' * SwToy * wPCA);

% Best of 1000 random directions, as a sanity floor.
jRandBest = 0;
for i = 1:1000
    wr = randn(2, 1);
    wr = wr / norm(wr);
    jr = (wr' * SbToy * wr) / (wr' * SwToy * wr);
    if jr > jRandBest
        jRandBest = jr;
    end
end

fprintf('  J(w_LDA)  = %.6f\n', jLDA);
fprintf('  J(w_PCA)  = %.6f\n', jPCA);
fprintf('  J(best of 1000 random) = %.6f\n', jRandBest);
fprintf('  ratio J(LDA)/J(PCA) = %.1f\n', jLDA / jPCA);

% ---- classification error along each 1-D direction ----
% Project, then threshold half-way between the two projected class means.
errLDA = toyError(X1, X2, wLDA);
errPCA = toyError(X1, X2, wPCA);

% ---- theoretical Bayes error for equal-covariance Gaussians ----
% Pe = Phi(-Delta/2) with Delta^2 = (mu2-mu1)' * inv(Sigma) * (mu2-mu1).
% normcdf needs the Statistics Toolbox, so use the identity
%     Phi(-a) = 0.5 * erfc(a / sqrt(2))
% with erfc, which is built into plain MATLAB.
delta2 = (mu2 - mu1)' * (Sigma \ (mu2 - mu1));
delta  = sqrt(delta2);
bayesErr = 0.5 * erfc((delta / 2) / sqrt(2));

fprintf('  error along w_LDA = %.2f%%\n', 100 * errLDA);
fprintf('  error along w_PCA = %.2f%%\n', 100 * errPCA);
fprintf('  Delta = %.4f, theoretical Bayes error = %.2f%%\n', delta, 100 * bayesErr);

% ---- BRUTE-FORCE CHECK THAT THE CLOSED FORM REALLY MAXIMISES J ----
% In 2-D every unit direction is (cos t, sin t) for some angle t, so we can
% simply try all of them on a fine grid and see which one gives the largest
% Fisher criterion.  If our formula w = inv(S_W)*(m2-m1) is right, the best
% angle on the grid must agree with it.  This is a direct numerical proof of
% the derivation and -- unlike fitcdiscr -- it needs no toolbox.
bestJ = -1;
bestT = 0;
for t = 0:0.0005:pi
    wt = [cos(t); sin(t)];
    jt = (wt' * SbToy * wt) / (wt' * SwToy * wt);
    if jt > bestJ
        bestJ = jt;
        bestT = t;
    end
end
wBrute = [cos(bestT); sin(bestT)];
fprintf('  brute-force best J over 6283 directions = %.6f\n', bestJ);
fprintf('  J(w_LDA) - J(brute) = %.3e  (should be >= 0, up to grid step)\n', jLDA - bestJ);
fprintf('  |cos angle| between closed-form w and brute-force w = %.8f\n', abs(wBrute' * wLDA));

% ---- optional cross-check against the Statistics Toolbox ----
% Not installed on this machine (only MATLAB + Image Processing + Computer
% Vision), so this block is skipped; the brute-force check above covers it.
if exist('fitcdiscr', 'file') == 2
    mdl = fitcdiscr(Xtoy', ytoy');
    % For 2 classes fitcdiscr stores the boundary as Const + Linear'*x = 0,
    % so the normal of that boundary is the discriminant direction.
    wML = mdl.Coeffs(1, 2).Linear;
    wML = wML / norm(wML);
    % Eigenvector / normal sign is arbitrary, so compare |cos(angle)|.
    cosAng = abs(wML' * wLDA);
    fprintf('  |cos angle| between our w and fitcdiscr = %.8f\n', cosAng);
else
    fprintf('  fitcdiscr not available, skipping toolbox cross-check\n');
end

% ---- figure: scatter with both directions ----
figure('Visible', 'off', 'Position', [100 100 900 400]);
subplot(1, 2, 1);
plot(X1(1, :), X1(2, :), '.', 'Color', [0.2 0.4 0.8]); hold on;
plot(X2(1, :), X2(2, :), '.', 'Color', [0.85 0.33 0.1]);
sc = 12;
plot([-sc * wPCA(1), sc * wPCA(1)], [-sc * wPCA(2), sc * wPCA(2)], 'k-', 'LineWidth', 2);
plot([-sc * wLDA(1), sc * wLDA(1)], [-sc * wLDA(2), sc * wLDA(2)], 'g-', 'LineWidth', 2);
axis equal; grid on;
xlabel('x'); ylabel('y');
title('Toy data: PCA (black) vs LDA (green)');
legend('class 1', 'class 2', 'PCA dir', 'LDA dir', 'Location', 'best');

subplot(2, 2, 2);
histogram(wLDA' * X1, 40); hold on; histogram(wLDA' * X2, 40);
title(sprintf('projected on w_{LDA}: error %.2f%%', 100 * errLDA));
grid on;

subplot(2, 2, 4);
histogram(wPCA' * X1, 40); hold on; histogram(wPCA' * X2, 40);
title(sprintf('projected on w_{PCA}: error %.2f%%', 100 * errPCA));
grid on;

saveas(gcf, [figDir 's1_toy.png']);

%% ====================================================================
%% S2  BUILD THE FOLDER LISTS FOR THE TWO FACE DATABASES
%
%  ORL  : subjects live in  ../ORL/s1 ... ../ORL/s40, 10 images each.
%         NOTE: the ORL directory also contains a stray empty folder called
%         'sed'.  X glob like dir('../ORL/s*') would pick it up and break the
%         loader.  We avoid the problem completely by BUILDING the path from
%         the subject index, exactly as the HW3 code did.
%
%  Yale : subjects live in ../CroppedYale/yaleB01 ... yaleB39, with yaleB14
%         missing (38 subjects).  The number of images per subject is NOT
%         constant -- it varies from 60 to 64 -- so nothing may assume 64.
%% ====================================================================

orlFolders40 = cell(1, 40);
for p = 1:40
    orlFolders40{p} = [orlRoot 's' num2str(p) '/'];
end
orlFolders32 = orlFolders40(1:32);          % the 32 subjects HW3 used

% For Yale we must list the directory, because yaleB14 does not exist and so
% the names are not a contiguous numeric range.  dir returns entries in
% alphabetical order, and for these names alphabetical order equals numeric
% order ('yaleB01' < 'yaleB02' < ... < 'yaleB39').
dy = dir(yaleRoot);
yaleFolders = {};
for i = 1:numel(dy)
    % keep real sub-directories only: skip '.', '..' and any hidden entry
    if dy(i).isdir && dy(i).name(1) ~= '.'
        yaleFolders{end+1} = [yaleRoot dy(i).name '/'];   %#ok<SAGROW>
    end
end
fprintf('\n--- S2: datasets found ---\n');
fprintf('  ORL subjects  : %d\n', numel(orlFolders40));
fprintf('  Yale subjects : %d\n', numel(yaleFolders));

%% ====================================================================
%% S3 / S4  EIGENFACES BASELINE -- REPRODUCE THE HW3 NUMBERS EXACTLY
%
%  This is a REGRESSION TEST.  The loader has been rewritten (preallocation
%  instead of growing the matrix inside the loop, one shared function instead
%  of copy-pasted ORL/Yale blocks), so we must prove the rewrite did not
%  change the data.  If these printed rates stop matching q1_matlab/run_log.txt
%  then the split logic has drifted and nothing downstream can be trusted.
%
%  Expected from HW3 (run_log.txt):
%    ORL  all coeffs @ ks_d : 17.2 44.5 71.1 75.8 93.0 92.2 92.2 94.5 95.3 ...
%    Yale all coeffs @ ks_d :  2.9  3.2  3.1  6.7 16.8 20.8 24.2 28.0 30.8 ...
%% ====================================================================

fprintf('\n--- S3/S4: eigenfaces baseline (HW3 regression check) ---\n');

ks_d = [1 2 3 5 10 15 20 30 50 75 100 150 170];

% ---------- ORL, 32 subjects, 6 train / 4 test, HW3 file order ----------
[imTrO, ytrO, imTeO, yteO] = loadFaces(orlFolders32, 6, 'hw3', dORL);
fprintf('  ORL-32 train %d x %d, test %d x %d\n', size(imTrO), size(imTeO));

[VO, lambdaO, xbarO] = doPCA(imTrO);
XO   = imTrO - xbarO;
XteO = imTeO - xbarO;

rateO = zeros(size(ks_d));
for t = 1:numel(ks_d)
    k = ks_d(t);
    Vk = VO(:, 1:k);
    rateO(t) = nnClassify(Vk' * XO, ytrO, Vk' * XteO, yteO);
end
fprintf('  ORL-32 eigenfaces : ');
fprintf('%.1f ', rateO);
fprintf('\n');

% ---------- Yale, 38 subjects, 40 train, HW3 file order ----------
[imTrY, ytrY, imTeY, yteY] = loadFaces(yaleFolders, 40, 'hw3', dYALE);
fprintf('  Yale   train %d x %d, test %d x %d\n', size(imTrY), size(imTeY));

[VY, lambdaY, xbarY] = doPCA(imTrY);
XY   = imTrY - xbarY;
XteY = imTeY - xbarY;

rateY = zeros(size(ks_d));
for t = 1:numel(ks_d)
    k = ks_d(t);
    Vk = VY(:, 1:k);
    rateY(t) = nnClassify(Vk' * XY, ytrY, Vk' * XteY, yteY);
end
fprintf('  Yale   eigenfaces : ');
fprintf('%.1f ', rateY);
fprintf('\n');

% ---------- the reconstruction-error identity ----------
% THEOREM: the mean squared reconstruction error when we keep k components
% equals the sum of the DISCARDED eigenvalues of the covariance matrix.
% Here lambda already contains eigenvalues of C = X*X'/M (doPCA divides the
% eigenvalues of L = X'*X by M), which is why the two columns agree.
fprintf('\n  Reconstruction error identity (ORL-32):\n');
fprintf('      k    measured error     sum of discarded eigenvalues\n');
for k = [10 50 100]
    Vk   = VO(:, 1:k);
    Xrec = xbarO + Vk * (Vk' * XO);
    errM = mean(sum((Xrec - imTrO).^2, 1));
    errT = sum(lambdaO(k+1:end));
    fprintf('    %4d  %16.4f  %16.4f\n', k, errM, errT);
end

%% ====================================================================
%% S5  THE SMALL-SAMPLE-SIZE PROBLEM, SHOWN NUMERICALLY
%
%  Fisher's LDA needs inv(S_W).  For face images that is impossible to do
%  directly, for two separate reasons, and we show BOTH here:
%
%   (1) SIZE.   S_W is d x d.  For Yale d = 32256, so S_W would hold
%       32256^2 = 1.04e9 doubles = 8.3 GB.  We never build it.
%   (2) RANK.   rank(S_W) <= N - c.  For Yale that is 1520 - 38 = 1482,
%       far below d = 32256, so S_W is singular no matter how it is built.
%
%  We verify (2) without ever forming the d x d matrix: S_W = Aw*Aw' where
%  Aw holds the within-class-centred images, and rank(Aw*Aw') = rank(Aw),
%  which we get from the small N x N Gram matrix Aw'*Aw.
%% ====================================================================

fprintf('\n--- S5: why S_W cannot be inverted in pixel space ---\n');

NY = size(imTrY, 2);
cY = numel(yaleFolders);

% Build the within-class-centred data matrix: subtract each class mean from
% the images of that class.
DwY = zeros(size(imTrY));
for i = 1:cY
    cols = find(ytrY == i);
    mi   = mean(imTrY(:, cols), 2);
    DwY(:, cols) = imTrY(:, cols) - mi;
end

LwY = DwY' * DwY;                 % N x N Gram matrix, cheap
rankSwY = rank(LwY);              % rank(S_W) = rank(Aw) = rank(Aw'*Aw)

fprintf('  Yale: d = %d, N = %d, c = %d\n', dYALE, NY, cY);
fprintf('  S_W in pixel space would be %d x %d = %.2f GB\n', ...
        dYALE, dYALE, (dYALE^2 * 8) / 1e9);
fprintf('  theory: rank(S_W) <= N - c = %d\n', NY - cY);
fprintf('  measured rank(S_W)         = %d\n', rankSwY);
fprintf('  so S_W is singular: rank %d << d %d\n', rankSwY, dYALE);

% ---- does the measured rank actually meet the bound N - c? ----
% N - c is an UPPER bound, so it is worth checking whether we reach it.
% We do now (1482 = 1482), but we did NOT at first: the dataset on this
% machine used to contain a stray duplicated file,
%     CroppedYale/yaleB12/'yaleB12_P00A+020E+10 - Copy.pgm',
% byte-identical to yaleB12_P00A+020E+10.pgm.  Two identical columns stay
% identical after within-class centring, so they were an exact linear
% dependency and cost exactly one unit of rank, giving 1481.  Deleting the
% stray file restored the bound.  This search is kept as a regression test:
% a rank one short of N - c is a dataset defect, not a theory failure.
fprintf('\n  Searching the training set for exact duplicate images...\n');
nDup = 0;
for i = 1:cY
    cols = find(ytrY == i);
    for u = 1:numel(cols)
        for v = (u+1):numel(cols)
            if norm(imTrY(:, cols(u)) - imTrY(:, cols(v))) == 0
                fprintf('    duplicate: subject %d, training images %d and %d\n', i, u, v);
                nDup = nDup + 1;
            end
        end
    end
end
fprintf('  exact duplicate pairs found = %d\n', nDup);
fprintf('  this accounts for the gap: %d (bound) - %d (duplicates) = %d (measured)\n', ...
        NY - cY, nDup, rankSwY);

clear DwY LwY;                    % free the memory again

%% ====================================================================
%% S6 / S7 / S8  MAIN COMPARISON
%
%  Four methods, exactly the ones the paper compares:
%
%   1. Correlation  -- nearest neighbour on raw pixels, after normalising
%                      each image to zero mean and unit norm.  No subspace.
%   2. Eigenfaces   -- PCA, nearest neighbour in the k-dimensional space.
%   3. Eigenfaces w/o first 3 -- same, but drop principal components 1..3,
%                      which (by the Lambertian argument) mostly encode the
%                      lighting direction rather than identity.
%   4. Fisherfaces  -- PCA down to N-c dimensions, then LDA, keeping c-1
%                      directions.
%
%  FAIRNESS NOTE: Fisherfaces always uses exactly c-1 features.  Quoting
%  Eigenfaces at its best k over all k would compare 50 or 170 features
%  against c-1, which is not a like-for-like comparison.  So we report
%  Eigenfaces BOTH at its best k AND at k = c-1, the matched-dimension case.
%% ====================================================================

fprintf('\n--- S6/S7/S8: main comparison ---\n');

% Each row: name, folders, numTrain, splitMode, d
configs = { ...
    'ORL-32  (HW3 split)',  {orlFolders32}, 6,  'hw3',    dORL,  orlH, orlW; ...
    'ORL-40  (HW3 split)',  {orlFolders40}, 6,  'hw3',    dORL,  orlH, orlW; ...
    'ORL-32  (numeric)',    {orlFolders32}, 6,  'numeric',dORL,  orlH, orlW; ...
    'Yale-38 (HW3 split)',  {yaleFolders},  40, 'hw3',    dYALE, yalH, yalW; ...
    'Yale-38 (random)',     {yaleFolders},  40, 'random', dYALE, yalH, yalW };

ksAll = [1 2 3 5 10 15 20 30 50 75 100 150 170];

nCfg = size(configs, 1);
results = zeros(nCfg, 5);   % corr | eig best | eig@(c-1) | eigDrop3 best | fisher
resEigBestK = zeros(nCfg, 1);

for ci = 1:nCfg
    name      = configs{ci, 1};
    folders   = configs{ci, 2}{1};
    numTrain  = configs{ci, 3};
    splitMode = configs{ci, 4};
    dPix      = configs{ci, 5};

    fprintf('\n  == %s ==\n', name);

    [imTr, ytr, imTe, yte] = loadFaces(folders, numTrain, splitMode, dPix);
    N = size(imTr, 2);
    c = numel(folders);
    fprintf('     N = %d, c = %d, d = %d, test = %d\n', N, c, dPix, size(imTe, 2));

    % ---- 1. correlation baseline (raw pixels) ----
    rCorr = nnClassify(normCols(imTr), ytr, normCols(imTe), yte);

    % ---- PCA once, reused by methods 2, 3 and 4 ----
    [V, lam, xbar] = doPCA(imTr);
    X   = imTr - xbar;
    Xte = imTe - xbar;

    % ---- 2 and 3. eigenfaces, with and without the first 3 components ----
    rEig  = zeros(1, numel(ksAll));
    rEig3 = nan(1, numel(ksAll));
    for t = 1:numel(ksAll)
        k = ksAll(t);
        if k > size(V, 2)
            rEig(t) = NaN;
            continue;
        end
        Vk = V(:, 1:k);
        P  = Vk' * X;
        Q  = Vk' * Xte;
        rEig(t) = nnClassify(P, ytr, Q, yte);
        if k > 3
            % keep rows 4..k, i.e. throw away the top 3 eigencoefficients
            rEig3(t) = nnClassify(P(4:k, :), ytr, Q(4:k, :), yte);
        end
    end

    % eigenfaces at the SAME number of features Fisherfaces gets
    Vcm1 = V(:, 1:(c-1));
    rEigMatched = nnClassify(Vcm1' * X, ytr, Vcm1' * Xte, yte);

    % ---- 4. fisherfaces ----
    m = N - c;                                  % the paper's choice
    [Wopt, info] = fisherFromPCA(V, xbar, imTr, ytr, m, 2);
    rFish = nnClassify(Wopt' * X, ytr, Wopt' * Xte, yte);

    [bEig, iEig] = max(rEig);
    bEig3 = max(rEig3);

    results(ci, :)  = [rCorr, bEig, rEigMatched, bEig3, rFish];
    resEigBestK(ci) = ksAll(iEig);

    fprintf('     correlation (raw pixels)        : %.1f%%\n', rCorr);
    fprintf('     eigenfaces, best k = %d          : %.1f%%\n', ksAll(iEig), bEig);
    fprintf('     eigenfaces at k = c-1 = %d       : %.1f%%\n', c-1, rEigMatched);
    fprintf('     eigenfaces w/o top 3, best      : %.1f%%\n', bEig3);
    fprintf('     FISHERFACES (%d features)        : %.1f%%\n', c-1, rFish);
    fprintf('     [S_W: m = %d, rank = %d, cond = %.3e, regularised = %d]\n', ...
            m, info.rankSw, info.condSw, info.usedReg);
    fprintf('     [check ||S_T - (S_B+S_W)|| / ||S_T|| = %.3e]\n', info.resST);
    fprintf('     [cross-check vs eig(S_B,S_W,''chol''): %.3e]\n', info.crossCheck);

    % keep the ORL-32 and Yale HW3-split pieces for the later sections
    % Keep only what the later sections actually need.  Holding the full
    % Yale matrices (X is 392 MB, V is another 392 MB) would waste over a
    % gigabyte for no reason, so we store a few sample images and the first
    % 60 basis vectors rather than everything.
    if ci == 1
        keepORL = struct('V', V, 'xbar', xbar, 'imTr', imTr, 'ytr', ytr, ...
                         'imTe', imTe, 'yte', yte, 'Wopt', Wopt, 'lam', lam, ...
                         'rEig', rEig, 'rEig3', rEig3, 'c', c, 'N', N);
    elseif ci == 4
        keepYale = struct('V', V(:, 1:60), 'xbar', xbar, 'sample', imTr(:, 1), ...
                          'Wopt', Wopt, 'lam', lam, ...
                          'rEig', rEig, 'rEig3', rEig3, 'c', c, 'N', N);
    end
    clear V X Xte imTr imTe Wopt;
end

% ---- summary table ----
fprintf('\n  ================ SUMMARY (recognition rate %%) ================\n');
fprintf('  %-22s %7s %9s %9s %9s %9s\n', 'dataset', 'corr', 'eig best', 'eig c-1', 'eig -top3', 'FISHER');
for ci = 1:nCfg
    fprintf('  %-22s %7.1f %9.1f %9.1f %9.1f %9.1f\n', configs{ci,1}, results(ci, :));
end
fprintf('  ==============================================================\n');

%% ====================================================================
%% S9  SWEEP THE PCA DIMENSION m USED BEFORE LDA
%
%  The paper fixes m = N - c.  The reason is a rank argument: rank(S_W) <=
%  N - c, so keeping more than N - c principal components guarantees a
%  singular S_W, while keeping exactly N - c makes it (generically) full
%  rank.
%
%  But "full rank" is not the same as "well conditioned".  At m = N - c the
%  rank of S_W is EXACTLY m, i.e. it is nonsingular with zero slack, so its
%  smallest eigenvalues are built from almost no data and are essentially
%  estimation noise.  Inverting then amplifies that noise.  This sweep asks
%  whether a smaller m -- throwing away those noisy directions -- actually
%  generalises better.
%% ====================================================================

fprintf('\n--- S9: sweep of PCA dimension m before LDA ---\n');

sweepCfgs = { ...
    'ORL-32', orlFolders32, 6,  'hw3',    dORL,  [31 40 60 80 100 120 140 160]; ...
    'Yale-38', yaleFolders, 40, 'random', dYALE, [37 60 100 200 400 700 1000 1200 1482] };

mSweepStore = cell(2, 1);

for si = 1:size(sweepCfgs, 1)
    nameS   = sweepCfgs{si, 1};
    folders = sweepCfgs{si, 2};
    nTr     = sweepCfgs{si, 3};
    sMode   = sweepCfgs{si, 4};
    dPix    = sweepCfgs{si, 5};
    mList   = sweepCfgs{si, 6};

    [imTr, ytr, imTe, yte] = loadFaces(folders, nTr, sMode, dPix);
    N = size(imTr, 2);
    c = numel(folders);
    [V, ~, xbar] = doPCA(imTr);
    X   = imTr - xbar;
    Xte = imTe - xbar;

    fprintf('\n  == %s ==  (N = %d, c = %d, N-c = %d)\n', nameS, N, c, N - c);
    fprintf('      m      rate%%     rank(S_W)   cond(S_W)    ridge?\n');

    rateM = zeros(size(mList));
    condM = zeros(size(mList));
    regM  = zeros(size(mList));
    for t = 1:numel(mList)
        m = mList(t);
        [W, inf1] = fisherFromPCA(V, xbar, imTr, ytr, m, 1);
        rateM(t) = nnClassify(W' * X, ytr, W' * Xte, yte);
        condM(t) = inf1.condSw;
        regM(t)  = inf1.usedReg;
        fprintf('   %5d   %7.1f   %9d   %.3e      %d\n', ...
                m, rateM(t), inf1.rankSw, inf1.condSw, inf1.usedReg);
    end

    [bestR, bestI] = max(rateM);
    fprintf('    best m = %d at %.1f%%  (paper would use m = N-c = %d, giving %.1f%%)\n', ...
            mList(bestI), bestR, N - c, rateM(end));

    mSweepStore{si} = struct('name', nameS, 'mList', mList, 'rate', rateM, ...
                             'cond', condM, 'Nmc', N - c);

    clear V X Xte imTr imTe;
end

% ---- figure ----
figure('Visible', 'off', 'Position', [100 100 950 400]);
for si = 1:2
    S = mSweepStore{si};
    subplot(1, 2, si);
    yyaxis left;
    plot(S.mList, S.rate, '-o', 'LineWidth', 1.5);
    ylabel('recognition rate (%)');
    yyaxis right;
    semilogy(S.mList, S.cond, '-s', 'LineWidth', 1.2);
    ylabel('cond(S_W)');
    xlabel('m = PCA dimension kept before LDA');
    title([S.name ':  N-c = ' num2str(S.Nmc)]);
    grid on;
end
saveas(gcf, [figDir 's9_m_sweep.png']);

%% ====================================================================
%% S10  SWEEP THE NUMBER OF TRAINING IMAGES PER SUBJECT
%
%  rank(S_W) = sum over classes of (n_i - 1).  Each subject contributes only
%  n_i - 1 directions to the within-class scatter, so with n_i = 2 we are
%  estimating the within-class covariance from ONE direction per person.
%  This is the small-sample-size problem, and it is exactly why Fisherfaces
%  struggles on ORL (6 images per subject = 5 directions each) while winning
%  comfortably on Yale (40 images per subject = 39 directions each).
%
%  We use the RANDOM Yale split here, so that the lighting bias of the
%  lexicographic split does not confound the training-size effect.
%% ====================================================================

fprintf('\n--- S10: sweep of training images per subject (Yale, random split) ---\n');

nList = [2 3 5 10 20 32 40];
rateFishN = zeros(size(nList));
rateEigN  = zeros(size(nList));
rateCorrN = zeros(size(nList));

fprintf('   n/subj     N    N-c   corr%%   eig(best)%%   fisher%%\n');
for t = 1:numel(nList)
    nTr = nList(t);
    [imTr, ytr, imTe, yte] = loadFaces(yaleFolders, nTr, 'random', dYALE);
    N = size(imTr, 2);
    c = numel(yaleFolders);

    rateCorrN(t) = nnClassify(normCols(imTr), ytr, normCols(imTe), yte);

    [V, ~, xbar] = doPCA(imTr);
    X   = imTr - xbar;
    Xte = imTe - xbar;

    % eigenfaces: best over a few k that are valid for this N
    ksHere = [10 20 30 50 75 100 150 170];
    ksHere = ksHere(ksHere <= size(V, 2));
    bestE = 0;
    for kk = ksHere
        Vk = V(:, 1:kk);
        r  = nnClassify(Vk' * X, ytr, Vk' * Xte, yte);
        if r > bestE
            bestE = r;
        end
    end
    rateEigN(t) = bestE;

    m = N - c;
    W = fisherFromPCA(V, xbar, imTr, ytr, m, 0);
    rateFishN(t) = nnClassify(W' * X, ytr, W' * Xte, yte);

    fprintf('   %5d  %5d  %5d  %6.1f  %10.1f  %8.1f\n', ...
            nTr, N, N - c, rateCorrN(t), rateEigN(t), rateFishN(t));

    clear V X Xte imTr imTe;
end

figure('Visible', 'off');
plot(nList, rateCorrN, '-^', nList, rateEigN, '-o', nList, rateFishN, '-s', ...
     'LineWidth', 1.5);
grid on;
xlabel('training images per subject');
ylabel('recognition rate (%)');
title('Yale (random split): effect of training-set size');
legend('correlation', 'eigenfaces (best k)', 'fisherfaces', 'Location', 'southeast');
saveas(gcf, [figDir 's10_n_sweep.png']);

%% ====================================================================
%% S11  PICTURES: MEAN FACE, EIGENFACES, FISHERFACES, RECONSTRUCTION
%
%  Eigenfaces look like smooth, ghostly faces: they are the directions of
%  largest variance, and for face data that variance is dominated by broad
%  lighting and shape changes.
%  Fisherfaces look noisier and much higher frequency.  That is expected,
%  not a bug: they are optimised to separate identities, not to reconstruct
%  images, so they emphasise the small localised differences between people
%  and actively suppress the big lighting directions.
%% ====================================================================

fprintf('\n--- S11: figures ---\n');

% ---------- ORL: mean face ----------
figure('Visible', 'off');
imagesc(reshape(keepORL.xbar, orlH, orlW));
colormap gray; axis image off;
title('ORL mean face');
saveas(gcf, [figDir 's11_orl_meanface.png']);

% ---------- ORL: first 25 eigenfaces ----------
figure('Visible', 'off', 'Position', [100 100 700 760]);
for i = 1:25
    subplot(5, 5, i);
    imagesc(reshape(keepORL.V(:, i), orlH, orlW));
    colormap gray; axis image off;
    title(['u_{' num2str(i) '}'], 'FontSize', 8);
end
sgtitle('ORL: first 25 eigenfaces');
saveas(gcf, [figDir 's11_orl_eigenfaces.png']);

% ---------- ORL: first 25 fisherfaces ----------
figure('Visible', 'off', 'Position', [100 100 700 760]);
for i = 1:min(25, size(keepORL.Wopt, 2))
    subplot(5, 5, i);
    imagesc(reshape(keepORL.Wopt(:, i), orlH, orlW));
    colormap gray; axis image off;
    title(['w_{' num2str(i) '}'], 'FontSize', 8);
end
sgtitle('ORL: first 25 fisherfaces');
saveas(gcf, [figDir 's11_orl_fisherfaces.png']);

% ---------- ORL: reconstruction as k grows ----------
x   = keepORL.imTr(:, 1);
ksR = [2 10 20 50 75 100 125 150 175];
figure('Visible', 'off', 'Position', [100 100 1000 450]);
subplot(2, 5, 1);
imagesc(reshape(x, orlH, orlW)); colormap gray; axis image off; title('original');
for t = 1:numel(ksR)
    Vk   = keepORL.V(:, 1:ksR(t));
    xRec = keepORL.xbar + Vk * (Vk' * (x - keepORL.xbar));
    subplot(2, 5, t + 1);
    imagesc(reshape(xRec, orlH, orlW)); colormap gray; axis image off;
    title(['k = ' num2str(ksR(t))]);
end
sgtitle('ORL: PCA reconstruction of one face');
saveas(gcf, [figDir 's11_orl_reconstruction.png']);

% ---------- Yale: eigenfaces vs fisherfaces side by side ----------
% 2 rows x 8 columns of 168x192 images needs a figure aspect ratio of about
% (8*168)/(2*192) = 3.5, otherwise MATLAB pads the gap between the rows with
% a lot of empty space.
figure('Visible', 'off', 'Position', [100 100 1200 400]);
for i = 1:8
    subplot(2, 8, i);
    imagesc(reshape(keepYale.V(:, i), yalH, yalW));
    colormap gray; axis image off;
    title(['u_{' num2str(i) '}'], 'FontSize', 8);
end
for i = 1:8
    subplot(2, 8, 8 + i);
    imagesc(reshape(keepYale.Wopt(:, i), yalH, yalW));
    colormap gray; axis image off;
    title(['w_{' num2str(i) '}'], 'FontSize', 8);
end
sgtitle('Yale: top row = eigenfaces, bottom row = fisherfaces');
saveas(gcf, [figDir 's11_yale_eigen_vs_fisher.png']);

% ---------- recognition-rate curves ----------
figure('Visible', 'off', 'Position', [100 100 950 400]);
subplot(1, 2, 1);
plot(ksAll, keepORL.rEig, '-o', ksAll, keepORL.rEig3, '-s', 'LineWidth', 1.4);
hold on;
yline(results(1, 5), 'g--', 'LineWidth', 1.8);
grid on; xlabel('k'); ylabel('recognition rate (%)');
title('ORL-32');
legend('eigenfaces', 'eigenfaces w/o top 3', 'fisherfaces (c-1)', 'Location', 'southeast');

subplot(1, 2, 2);
plot(ksAll, keepYale.rEig, '-o', ksAll, keepYale.rEig3, '-s', 'LineWidth', 1.4);
hold on;
yline(results(4, 5), 'g--', 'LineWidth', 1.8);
grid on; xlabel('k'); ylabel('recognition rate (%)');
title('Yale-38 (HW3 split)');
legend('eigenfaces', 'eigenfaces w/o top 3', 'fisherfaces (c-1)', 'Location', 'southeast');
saveas(gcf, [figDir 's11_rate_curves.png']);

fprintf('  figures written to %s\n', figDir);

%% ====================================================================
%% S12  NUMERICAL CHECKS
%
%  Four independent things are verified here:
%   (a) the eigenfaces really are orthonormal;
%   (b) the L = X'*X route agrees with a direct SVD of X;
%   (c) Fisherfaces are S_W-orthogonal but NOT Euclidean-orthogonal;
%   (d) the number of useful Fisherfaces is capped at c-1 because
%       rank(S_B) <= c-1.
%% ====================================================================

fprintf('\n--- S12: numerical checks ---\n');

% ---- (a) orthonormality of the eigenfaces ----
V20 = keepORL.V(:, 1:20);
fprintf('  (a) ||V''V - I||_F for 20 ORL eigenfaces = %.3e\n', ...
        norm(V20' * V20 - eye(20), 'fro'));

% ---- (b) eig(X''X) route vs direct svd(X) ----
% The two give the same subspace, but svd works on X directly while the Gram
% matrix X'*X squares the condition number, so svd is the better-conditioned
% route.  Eigenvector SIGN is arbitrary, so we compare |u1'*u2|, not u1-u2.
Aorl = keepORL.imTr - keepORL.xbar;
[Vsvd, ~, ~] = svd(Aorl, 'econ');
agree = zeros(1, 10);
for i = 1:10
    agree(i) = abs(Vsvd(:, i)' * keepORL.V(:, i));
end
fprintf('  (b) |u_svd . u_eig| for the first 10 components:\n      ');
fprintf('%.6f ', agree);
fprintf('\n');
fprintf('      min over those 10 = %.8f (1.0 means identical direction)\n', min(agree));

% ---- (c) Fisherfaces are S_W-orthogonal, not Euclidean-orthogonal ----
% Rebuild the ORL scatter matrices in the PCA space to make the point.
mORL = keepORL.N - keepORL.c;
Vm   = keepORL.V(:, 1:mORL);
Alpha    = Vm' * Aorl;
alphaBar  = mean(Alpha, 2);
SwO  = zeros(mORL, mORL);
for i = 1:keepORL.c
    AlphaJ = Alpha(:, keepORL.ytr == i);
    mi = mean(AlphaJ, 2);
    Di = AlphaJ - mi;
    SwO = SwO + Di * Di';
end
SwO = (SwO + SwO') / 2;

Wf  = keepORL.Wopt;
WfY = Vm' * Wf;                       % the same directions, expressed in PCA space
G_eu = Wf' * Wf;                      % Euclidean Gram matrix
G_sw = WfY' * SwO * WfY;              % S_W-weighted Gram matrix

offEu = norm(G_eu - diag(diag(G_eu)), 'fro') / norm(G_eu, 'fro');
offSw = norm(G_sw - diag(diag(G_sw)), 'fro') / norm(G_sw, 'fro');
fprintf('  (c) off-diagonal energy of W''W      (Euclidean) = %.3e\n', offEu);
fprintf('      off-diagonal energy of W''S_W W  (S_W-inner) = %.3e\n', offSw);
fprintf('      => Fisherfaces are S_W-orthogonal, NOT Euclidean-orthogonal\n');

% ---- (d) the c-1 ceiling ----
SbO = zeros(mORL, mORL);
for i = 1:keepORL.c
    AlphaJ = Alpha(:, keepORL.ytr == i);
    ni = size(AlphaJ, 2);
    mi = mean(AlphaJ, 2);
    SbO = SbO + ni * (mi - alphaBar) * (mi - alphaBar)';
end
SbO = (SbO + SbO') / 2;
% ---- (e) the generalised eigenvalue spectrum of the pencil (S_B, S_W) ----
% Theory: the pencil has m eigenvalues, of which at most c-1 are non-zero
% (because rank(S_B) <= c-1).  The non-zero ones ARE the Fisher criterion
% values J(w_l).  Everything past index c-1 should be numerically zero.
mE   = keepORL.N - keepORL.c;
VmE  = keepORL.V(:, 1:mE);
AlE  = VmE' * (keepORL.imTr - keepORL.xbar);
abE  = mean(AlE, 2);
SwE  = zeros(mE, mE);
SbE  = zeros(mE, mE);
for j = 1:keepORL.c
    Aj = AlE(:, keepORL.ytr == j);
    nj = size(Aj, 2);
    mj = mean(Aj, 2);
    Dj = Aj - mj;
    SwE = SwE + Dj * Dj';
    SbE = SbE + nj * (mj - abE) * (mj - abE)';
end
SwE = (SwE + SwE') / 2;
SbE = (SbE + SbE') / 2;
RE  = chol(SwE);
ME  = (RE') \ (SbE / RE); ME = (ME + ME') / 2;
lamE = sort(eig(ME), 'descend');
fprintf('\n  (e) generalised eigenvalues of the pencil (S_B, S_W), ORL-32:\n');
fprintf('      m = %d eigenvalues in total, theory allows at most c-1 = %d non-zero\n', ...
        mE, keepORL.c - 1);
fprintf('      largest 3 : %.4f  %.4f  %.4f\n', lamE(1), lamE(2), lamE(3));
fprintf('      l = c-2, c-1 (last two that may be non-zero): %.4f  %.4f\n', ...
        lamE(keepORL.c-2), lamE(keepORL.c-1));
fprintf('      l = c  (first that must vanish)             : %.3e\n', lamE(keepORL.c));
fprintf('      largest |lambda| beyond index c-1           : %.3e\n', ...
        max(abs(lamE(keepORL.c:end))));
fprintf('      ratio lambda_1 / max|lambda_{l>=c}|         : %.3e\n', ...
        lamE(1) / max(abs(lamE(keepORL.c:end))));

fprintf('  (d) ORL-32: c = %d, so theory says rank(S_B) <= %d\n', keepORL.c, keepORL.c - 1);
fprintf('      measured rank(S_B) = %d, number of fisherfaces = %d\n', ...
        rank(SbO), size(keepORL.Wopt, 2));

% ---- (f) upper vs lower Cholesky: the same factorisation, written twice ----
% chol(Sw) returns the UPPER factor R, meaning R'*R = Sw.  That is what the main
% code uses.  Most textbooks instead write Sw = L*L' with L LOWER triangular.
% Claim: L is simply R transposed, so both routes must give the same Fisherfaces.
% We check that instead of believing it.  Reuses SwE and SbE built in part (e).

LE = chol(SwE, 'lower');          % lower factor, so that LE*LE' = SwE

% Build the whitened matrix M in both conventions.
M_up = (RE') \ (SbE / RE);        % upper:  M = inv(R')  * Sb * inv(R)
M_lo = LE    \ (SbE / LE');       % lower:  M = inv(L)   * Sb * inv(L')

% Both are symmetric, so plain eig is safe.  Sort largest eigenvalue first.
[Z_up, D_up] = eig((M_up + M_up') / 2);
[Z_lo, D_lo] = eig((M_lo + M_lo') / 2);
[lam_up, ord_up] = sort(diag(D_up), 'descend');
[lam_lo, ord_lo] = sort(diag(D_lo), 'descend');

% Keep the c-1 directions we would actually use, mapping each z back to a w.
nf   = keepORL.c - 1;
W_up = zeros(mE, nf);
W_lo = zeros(mE, nf);
for l = 1:nf
    z_up = Z_up(:, ord_up(l));    % l-th eigenvector from the upper route
    z_lo = Z_lo(:, ord_lo(l));    % l-th eigenvector from the lower route
    W_up(:, l) = RE  \ z_up;      % w = inv(R) *z, i.e. solve R *w = z
    W_lo(:, l) = LE' \ z_lo;      % w = inv(L')*z, i.e. solve L'*w = z
end

% Compare the two answers one direction at a time.  We take |cos(angle)| because
% an eigenvector's sign is arbitrary, so a value of 1 means "same direction".
worstCos = 1;
for l = 1:nf
    a   = W_up(:, l) / norm(W_up(:, l));
    b   = W_lo(:, l) / norm(W_lo(:, l));
    cosAngle = abs(a' * b);
    if cosAngle < worstCos
        worstCos = cosAngle;
    end
end

% Report RELATIVE differences.  S_B here has entries of order 1e7, so an
% absolute norm of 1e-4 would look alarming while actually being round-off.
% Dividing by the size of the thing we are comparing makes the number readable.
fprintf('\n  (f) upper (R''R) vs lower (LL'') Cholesky, ORL-32, m = %d:\n', mE);
fprintf('      ||L - R''|| / ||R||                 : %.3e\n', ...
        norm(LE - RE') / norm(RE));
fprintf('      ||M_up - M_lo|| / ||M_up||         : %.3e  <- same inner matrix\n', ...
        norm(M_up - M_lo) / norm(M_up));
fprintf('      max|lam_up - lam_lo| / lam_up(1)   : %.3e\n', ...
        max(abs(lam_up - lam_lo)) / lam_up(1));
fprintf('      worst |cos| between the %2d pairs   : %.8f  (1 = same direction)\n', ...
        nf, worstCos);
fprintf('      ||W''*S_W*W - I|| / sqrt(c-1)       : %.3e  <- S_W-orthogonality\n', ...
        norm(W_up' * SwE * W_up - eye(nf)) / sqrt(nf));
fprintf('      => the choice of convention is bookkeeping, not mathematics\n');

%% ====================================================================
%% S13  WHY PCA-THEN-LDA, AND NOT ONE OF THE OTHER FIXES?
%
%  S_W is singular, and there is more than one way to cope.  This section
%  measures the main ones side by side so the choice is defended with
%  numbers rather than opinion:
%
%   (a) the paper's recipe  : PCA to m = N-c, then plain LDA
%   (b) the same, with m tuned by the sweep of S9
%   (c) keep MORE components (m = N-1, where S_W is guaranteed singular)
%       and cope with the singularity by
%          - ridge / regularised LDA:  S_W + eps*I, for several eps
%          - the pseudo-inverse:       pinv(S_W)*S_B
%
%  The chol failure at m = N-1 is itself a result: it is the rank theorem
%  (rank(S_W) <= N-c) showing up as a numerical fact.
%% ====================================================================

fprintf('\n--- S13: alternative ways to handle a singular S_W ---\n');

altCfgs = { 'ORL-32 ', orlFolders32, 6,  'hw3',    dORL; ...
            'Yale-38', yaleFolders,  40, 'random', dYALE };

for ai = 1:size(altCfgs, 1)
    nameA   = altCfgs{ai, 1};
    folders = altCfgs{ai, 2};
    nTrA    = altCfgs{ai, 3};
    modeA   = altCfgs{ai, 4};
    dPixA   = altCfgs{ai, 5};

    rng(0);                                   % same split for every variant
    [imTr, ytr, imTe, yte] = loadFaces(folders, nTrA, modeA, dPixA);
    N = size(imTr, 2);
    c = numel(folders);
    [V, ~, xbar] = doPCA(imTr);
    X   = imTr - xbar;
    Xte = imTe - xbar;

    fprintf('\n  == %s ==  (N = %d, c = %d, N-c = %d, N-1 = %d)\n', ...
            nameA, N, c, N - c, N - 1);
    fprintf('    %-42s %8s %12s\n', 'method', 'rate%', 'cond(S_W)');

    % (a) the paper
    [r, cn] = ldaVariant(V, X, Xte, ytr, yte, c, N - c, 'chol', 0);
    fprintf('    %-42s %8.1f %12.2e\n', 'PCA to N-c, plain LDA   <-- the paper', r, cn);

    % (b) m tuned
    bestR = 0; bestM = 0;
    for m = round(linspace(c, N - c, 10))
        m = max(m, c);
        r2 = ldaVariant(V, X, Xte, ytr, yte, c, m, 'chol', 0);
        if ~isnan(r2) && r2 > bestR
            bestR = r2; bestM = m;
        end
    end
    fprintf('    %-42s %8.1f %12s\n', ...
            sprintf('PCA to tuned m = %d, plain LDA', bestM), bestR, '-');

    % (c) keep m = N-1, where S_W must be singular
    m2 = N - 1;
    [r3, cn3] = ldaVariant(V, X, Xte, ytr, yte, c, m2, 'chol', 0);
    if isnan(r3)
        fprintf('    %-42s %8s %12.2e\n', ...
                sprintf('PCA to N-1 = %d, plain LDA', m2), 'chol FAILS', cn3);
    else
        fprintf('    %-42s %8.1f %12.2e\n', ...
                sprintf('PCA to N-1 = %d, plain LDA', m2), r3, cn3);
    end
    for e = [1e-8 1e-6 1e-4 1e-2]
        r4 = ldaVariant(V, X, Xte, ytr, yte, c, m2, 'ridge', e);
        fprintf('    %-42s %8.1f %12s\n', ...
                sprintf('   ridge at m = N-1, eps = %.0e', e), r4, '-');
    end
    r5 = ldaVariant(V, X, Xte, ytr, yte, c, m2, 'pinv', 0);
    fprintf('    %-42s %8.1f %12s\n', '   pinv(S_W)*S_B at m = N-1', r5, '-');

    clear V X Xte imTr imTe;
end

%% ====================================================================
%% S14  EIGENFACES vs FISHERFACES: IS "HIGH FREQUENCY" MEASURABLE?
%
%  Section S11 shows that eigenfaces look smooth and Fisherfaces look
%  noisy.  That should not rest on an impression, and it is not merely an
%  observation either -- it is predictable.  Whitening maps direction l to
%  1/sqrt(lambda_l) times itself, so the SMALLEST-variance directions of
%  S_W are amplified most; and the smallest-variance within-class
%  directions are the fine-detail (high spatial frequency) ones.  So
%  Fisherfaces must come out high frequency.
%
%  We measure two things per basis image, via its 2-D FFT:
%     freq centroid = power-weighted mean radial spatial frequency
%     %hi           = share of power above a radial cutoff of 0.15
%  and we check the mechanism by measuring the frequency of S_W's own
%  eigen-directions as a function of their eigenvalue rank.
%% ====================================================================

fprintf('\n--- S14: frequency content of the two bases (ORL-32) ---\n');

Hs = orlH; Ws = orlW;
[fy, fx] = ndgrid((0:Hs-1)/Hs, (0:Ws-1)/Ws);
fy = min(fy, 1-fy); fx = min(fx, 1-fx);
radGrid = sqrt(fy.^2 + fx.^2);

nShow = 10;
ceV = zeros(1,nShow); heV = zeros(1,nShow);
cfV = zeros(1,nShow); hfV = zeros(1,nShow);
for l = 1:nShow
    [ceV(l), heV(l)] = specStats(keepORL.V(:, l),    Hs, Ws, radGrid);
    [cfV(l), hfV(l)] = specStats(keepORL.Wopt(:, l), Hs, Ws, radGrid);
end
fprintf('    l   eigenface: centroid  %%hi     fisherface: centroid  %%hi\n');
for l = 1:nShow
    fprintf('   %2d            %6.4f  %5.1f                 %6.4f  %5.1f\n', ...
            l, ceV(l), 100*heV(l), cfV(l), 100*hfV(l));
end
fprintf('   MEAN          %6.4f  %5.1f                 %6.4f  %5.1f\n', ...
        mean(ceV), 100*mean(heV), mean(cfV), 100*mean(hfV));
fprintf('   => Fisherfaces sit %.2fx higher in frequency than eigenfaces\n', ...
        mean(cfV)/mean(ceV));

% ---- the mechanism: S_W's small-eigenvalue directions are the fine ones ----
mO  = keepORL.N - keepORL.c;
VmO = keepORL.V(:, 1:mO);
AlO = VmO' * (keepORL.imTr - keepORL.xbar);
abO = mean(AlO, 2);
SwO = zeros(mO, mO);
for j = 1:keepORL.c
    Aj = AlO(:, keepORL.ytr == j);
    Dj = Aj - mean(Aj, 2);
    SwO = SwO + Dj * Dj';
end
SwO = (SwO + SwO') / 2;
[Qs, Ds] = eig(SwO);
[evS, si] = sort(diag(Ds), 'descend');
Qs = Qs(:, si);
fprintf('\n   S_W eigenvalues: largest %.3e, smallest %.3e, ratio %.2e\n', ...
        evS(1), evS(end), evS(1)/evS(end));
fprintf('   whitening scales direction l by 1/sqrt(lambda_l), so the smallest\n');
fprintf('   gets amplified ~%.0fx relative to the largest.\n', sqrt(evS(1)/evS(end)));
fprintf('   frequency centroid of S_W directions, by eigenvalue rank:\n');
for pick = [1 5 20 80 mO]
    v = VmO * Qs(:, pick);
    v = v / norm(v);
    cc = specStats(v, Hs, Ws, radGrid);
    fprintf('      rank %3d  (lambda = %.3e) : centroid %.4f\n', pick, evS(pick), cc);
end
fprintf('   => low rank (big lambda) is smooth, high rank (small lambda) is fine\n');
fprintf('      detail, so inverting S_W pushes energy into the fine directions.\n');

%% ====================================================================
%% S15  SIX ROUTES TO THE SAME GENERALISED EIGENPROBLEM
%
%  S_B w = lambda S_W w can be solved several ways.  We use Cholesky
%  whitening; this section checks that choice against the alternatives on
%  identical S_B and S_W, and reports speed, whether the route produces
%  complex eigenvalues, and whether the resulting (c-1)-dimensional
%  subspace actually differs (largest principal angle against ours).
%% ====================================================================

fprintf('\n--- S15: ways of solving S_B w = lambda S_W w ---\n');

rtCfgs = { 'ORL-32 ', orlFolders32, 6,  'hw3',    dORL; ...
           'Yale-38', yaleFolders,  40, 'random', dYALE };
rtNames = { '1 inv(Sw)*Sb, then eig', '2 Sw\Sb, then eig', ...
            '3 Cholesky whitening  <-- ours', '4 symmetric sqrt Sw^-1/2', ...
            '5 eig(Sb,Sw)  [QZ]', '6 eig(Sb,Sw,''chol'')' };

for ri = 1:size(rtCfgs, 1)
    rng(0);
    [imTr, ytr, imTe, yte] = loadFaces(rtCfgs{ri,2}, rtCfgs{ri,3}, rtCfgs{ri,4}, rtCfgs{ri,5});
    N = size(imTr, 2);
    c = numel(rtCfgs{ri,2});
    [V, ~, xbar] = doPCA(imTr);
    X = imTr - xbar;  Xte = imTe - xbar;
    m = N - c;
    Vm = V(:, 1:m);  Al = Vm' * X;  ab = mean(Al, 2);
    Sw = zeros(m, m); Sb = zeros(m, m);
    for j = 1:c
        Aj = Al(:, ytr == j); nj = size(Aj, 2); muj = mean(Aj, 2); Dj = Aj - muj;
        Sw = Sw + Dj * Dj';
        Sb = Sb + nj * (muj - ab) * (muj - ab)';
    end
    Sw = (Sw + Sw') / 2;  Sb = (Sb + Sb') / 2;
    fprintf('\n  == %s ==  m = %d, cond(S_W) = %.3e\n', rtCfgs{ri,1}, m, cond(Sw));
    fprintf('    %-32s %7s %8s %12s %12s\n', 'route', 'rate%', 'sec', 'max|imag|', 'angle vs #3');

    Wall = cell(1, 6); rt = zeros(1,6); ts = zeros(1,6); imx = zeros(1,6);
    for r = 1:6
        [Wall{r}, ts(r), imx(r)] = solveRoute(r, Sw, Sb, Vm, c);
        rt(r) = nnClassify(Wall{r}' * X, ytr, Wall{r}' * Xte, yte);
    end
    [Qa, ~] = qr(Wall{3}, 0);
    for r = 1:6
        [Qb, ~] = qr(Wall{r}, 0);
        sv = svd(Qa' * Qb);
        ang = acosd(min(1, max(-1, min(sv))));
        fprintf('    %-32s %7.1f %8.2f %12.2e %9.2f deg\n', ...
                rtNames{r}, rt(r), ts(r), imx(r), ang);
    end
    clear V X Xte imTr imTe Sw Sb Vm Al Wall;
end

fprintf('\n=====================================================\n');
fprintf('  DONE\n');
fprintf('=====================================================\n');

%% ====================================================================
%% LOCAL FUNCTIONS
%  MATLAB requires every local function of a script to appear AFTER all the
%  executable code, which is why they are all collected down here.
%% ====================================================================

function [imTr, ytr, imTe, yte] = loadFaces(folders, numTrain, splitMode, d)
% LOADFACES  Read PGM images, vectorise them, split into train and test.
%
%   folders   cell array of directory paths, one per subject (= per class)
%   numTrain  how many images of each subject go into the training set
%   splitMode 'hw3'     keep the order dir() returns (what HW3 did)
%             'numeric' sort by the number in the filename (ORL only)
%             'random'  shuffle
%   d         number of pixels per image
%
%   imTr is d x Ntrain: ONE IMAGE PER COLUMN.  ytr holds the class label.

nF = numel(folders);

% ---- pass 1: count files so we can preallocate ----
% HW3 grew the matrix with  X = [X, im(:)]  inside the loop, which makes
% MATLAB reallocate and copy the whole array on every single image.  For
% Yale that is 1520 copies of a 392 MB matrix.  Counting first and then
% filling a preallocated array removes that cost entirely.
nTrainTot = 0;
nTestTot  = 0;
for p = 1:nF
    f  = dir([folders{p} '*.pgm']);
    nf = numel(f);
    ntr = min(numTrain, nf);              % a subject may have fewer images
    nTrainTot = nTrainTot + ntr;
    nTestTot  = nTestTot  + (nf - ntr);
end

imTr = zeros(d, nTrainTot);
ytr = zeros(1, nTrainTot);
imTe = zeros(d, nTestTot);
yte = zeros(1, nTestTot);

% ---- pass 2: actually read the images ----
a = 0;
b = 0;
for p = 1:nF
    files = dir([folders{p} '*.pgm']);
    nf    = numel(files);

    order = 1:nf;                          % default: whatever dir() gave us
    if strcmp(splitMode, 'numeric')
        % ORL filenames are 1.pgm, 2.pgm, ..., 10.pgm.  dir() sorts them as
        % TEXT, giving 1,10,2,3,... so we re-sort by the actual number.
        vals = zeros(1, nf);
        for j = 1:nf
            vals(j) = sscanf(files(j).name, '%d');
        end
        [~, order] = sort(vals);
    elseif strcmp(splitMode, 'random')
        order = randperm(nf);
    end

    for j = 1:nf
        im = double(imread([folders{p} files(order(j)).name]));
        if j <= numTrain
            a = a + 1;
            imTr(:, a) = im(:);             % im(:) stacks the image COLUMN BY COLUMN
            ytr(a)    = p;
        else
            b = b + 1;
            imTe(:, b) = im(:);
            yte(b)    = p;
        end
    end
end
end


function [V, lambda, xbar] = doPCA(imgs)
% DOPCA  Principal components of the d x N image matrix imgs (one RAW image
%        per column), computed with the small-matrix trick L = X'*X.
%
%  Notation follows the course slides ("PCA: algorithm") and HW3 Q1:
%     xbar   d x 1   the mean face,  xbar = (1/N) sum_i x_i
%     X      d x N   the mean-deducted data, X = [xbar_1 | ... | xbar_N]
%     L      N x N   the small matrix X'*X
%     W      N x N   eigenvectors of L  (these are sir's w, in R^N)
%     V      d x N   eigenvectors of C  (the eigenfaces), unit norm,
%                    most important first
%     lambda         eigenvalues of C = X*X'/N, descending
%
%  WHY THE TRICK: the covariance C = X*X'/N is d x d.  For Yale d = 32256,
%  so C has 1.04e9 entries (about 8 GB) -- we cannot even store it.  But X
%  has only N columns, so L = X'*X is only N x N.  If L*w = lam*w then
%      X*X'*(X*w) = X*(X'*X*w) = X*(lam*w) = lam*(X*w),
%  so X*w is an eigenvector of X*X' with the SAME eigenvalue.

[~, N] = size(imgs);

xbar = mean(imgs, 2);
X    = imgs - xbar;              % implicit expansion: subtracts from every column
L    = X' * X;                   % N x N, cheap

% W holds the eigenvectors of the SMALL matrix L; these are sir's w, living
% in R^N.  Not to be confused with the Fisherfaces (also W) elsewhere in this
% file, which live in R^d -- the two never meet, and this W is local to doPCA.
[W, D] = eig(L);

% eig returns eigenvalues in ASCENDING order.  We want the largest variance
% directions first, so we must sort descending -- this is not optional.
[eigvals, idx] = sort(diag(D), 'descend');
W = W(:, idx);

V = X * W;                       % map back to d-dimensional image space

% X*w has norm sqrt(lam), not 1 (eig already returns w with norm 1), so every
% column must be rescaled to unit length.  Dividing a VECTOR by the SCALAR
% norm(V(:,l)) leaves its direction alone and makes its length exactly 1.
%
% After mean subtraction rank(X) <= N-1, so the last eigenvalue is numerically
% zero (on ORL-32 it comes out as -1.1e-06, i.e. slightly negative) and the
% corresponding column is pure round-off.  Its norm is tiny but NOT tiny in an
% absolute sense -- on ORL-32 it is 1.95e-09, while the leading column has norm
% 2.37e+04.  So the threshold has to be RELATIVE to the data scale; an absolute
% 1e-10 would never fire here and would happily blow a round-off column up to
% unit length.  We compare against the largest column norm instead.
scale = norm(V(:, 1));                 % largest column norm = sqrt(lambda_1)
for l = 1:size(V, 2)
    nrm = norm(V(:, l));
    if nrm > 1e-10 * scale
        V(:, l) = V(:, l) / nrm;
    else
        V(:, l) = 0;                   % numerically-zero direction: do not use
    end
end

% Eigenvalues of L are N times the eigenvalues of C = X*X'/N.
lambda = eigvals / N;
end


function rate = nnClassify(trainProj, trainLabels, testProj, testLabels)
% NNCLASSIFY  Nearest-neighbour classification in the projected space,
%             using squared Euclidean distance.  Returns a percentage.
%
%  We compare SQUARED distances and never take a square root: sqrt is
%  monotonic, so it cannot change which neighbour is closest, and skipping
%  it saves one operation per comparison.

correct = 0;
for i = 1:size(testProj, 2)
    % trainProj - testProj(:,i) subtracts the test vector from EVERY column
    % (implicit expansion); sum(...,1) then collapses each column to one
    % squared distance.
    distances = sum((trainProj - testProj(:, i)).^2, 1);
    [~, best] = min(distances);
    if trainLabels(best) == testLabels(i)
        correct = correct + 1;
    end
end
rate = 100 * correct / size(testProj, 2);
end


function err = toyError(X1, X2, w)
% TOYERROR  Project both classes onto direction w, put the threshold
%           half-way between the two projected means, count mistakes.
p1  = w' * X1;
p2  = w' * X2;
thr = 0.5 * (mean(p1) + mean(p2));
if mean(p1) < mean(p2)
    wrong = sum(p1 > thr) + sum(p2 <= thr);
else
    wrong = sum(p1 <= thr) + sum(p2 > thr);
end
err = wrong / (numel(p1) + numel(p2));
end


function Xn = normCols(X)
% NORMCOLS  Normalise every image (column) to zero mean and unit norm.
%
%  This is what makes nearest neighbour on raw pixels a CORRELATION
%  classifier: for unit-norm zero-mean vectors a and b,
%      ||a-b||^2 = 2 - 2*a'*b,
%  so the smallest Euclidean distance is exactly the largest correlation.
%  Normalising also removes a constant brightness offset and a global
%  contrast scaling from each image.

Xn = X - mean(X, 1);              % mean(X,1) is the per-COLUMN (per-image) mean
for i = 1:size(Xn, 2)
    nrm = norm(Xn(:, i));
    if nrm > 1e-10
        Xn(:, i) = Xn(:, i) / nrm;
    end
end
end


function [Wopt, info] = fisherFromPCA(V, xbar, imgs, y, m, verbose)
% FISHERFROMPCA  The Fisherfaces construction, stage 2.
%
%  Takes an ALREADY COMPUTED PCA basis V (so that sweeps over m do not
%  recompute the expensive eigendecomposition) and returns
%      Wopt = Vm * Wfld       (d x (c-1))
%  the full projection from pixel space straight to the discriminant space.
%
%  Inputs
%    V        d x N PCA basis (eigenfaces) from doPCA
%    xbar     d x 1 mean face
%    imgs, y  raw training images (one per column) and their class labels
%    m        how many principal components to keep before LDA
%             (the paper uses m = N - c)
%    verbose  0 = no diagnostics (fast, for sweeps)
%             1 = also compute rank(S_W) and cond(S_W)
%             2 = additionally cross-check against eig(S_B,S_W,'chol')
%
%  Index convention matches the study guide: i over images, l over
%  components, j over classes.

classes = unique(y);
c = numel(classes);

Vm    = V(:, 1:m);                % the eigenspace we work in
X     = imgs - xbar;              % mean-deducted images
Alpha = Vm' * X;                  % m x N : eigen-coefficients of every image

% ---- scatter matrices, built in the m-dimensional eigenspace ----
alphaBar = mean(Alpha, 2);
Sw = zeros(m, m);
Sb = zeros(m, m);
for j = 1:c
    AlphaJ = Alpha(:, y == classes(j));   % logical indexing picks class j's columns
    nj  = size(AlphaJ, 2);
    muj = mean(AlphaJ, 2);                % class mean, in eigenspace
    Dj  = AlphaJ - muj;
    Sw = Sw + Dj * Dj';                                 % within-class scatter
    Sb = Sb + nj * (muj - alphaBar) * (muj - alphaBar)'; % between-class scatter
end

% Both matrices are symmetric in exact arithmetic but floating point leaves
% tiny asymmetries, and chol/eig below assume exact symmetry.  Forcing it
% costs nothing and avoids spurious complex eigenvalues later.
Sw = (Sw + Sw') / 2;
Sb = (Sb + Sb') / 2;

% ---- free correctness check: S_T = S_B + S_W ----
AlphaC = Alpha - alphaBar;
St = AlphaC * AlphaC';
info.resST = norm(St - (Sb + Sw), 'fro') / norm(St, 'fro');

% ---- solve  S_B w = lambda S_W w  by Cholesky whitening ----
%  Write S_W = R'*R (Cholesky).  Substituting w = inv(R)*z:
%       S_B inv(R) z = lambda R'R inv(R) z = lambda R' z
%  and multiplying on the left by inv(R'):
%       inv(R') S_B inv(R) z = lambda z
%  which is an ORDINARY eigenproblem for a SYMMETRIC matrix.
%
%  Why bother?  Note first that S_W IS invertible here: we are in the reduced
%  m-dimensional PCA space with m <= N-c, not in pixel space.  So the obvious
%  alternative inv(S_W)*S_B can be formed, and S15 measures that it gives the
%  SAME answer (0.00 deg, same recognition rate).  We use Cholesky anyway
%  because: it is ~2.6x faster, it needs no real()/re-sort cleanup, and the
%  symmetric solver GUARANTEES real eigenvalues and S_W-orthogonal vectors
%  rather than happening to produce them.  inv(S_W)*S_B is not symmetric,
%  since its transpose is S_B*inv(S_W), so it carries no such guarantee.
[R, p] = chol(Sw);
info.usedReg = 0;
if p ~= 0
    % chol failed => S_W is not numerically positive definite.  Add a small
    % ridge, scaled to the size of S_W so it is dimensionally sensible.
    ridge = 1e-6 * trace(Sw) / m;
    Sw = Sw + ridge * eye(m);
    [R, p2] = chol(Sw);
    info.usedReg = 1;
    if p2 ~= 0
        error('fisherFromPCA: S_W still not positive definite after ridge');
    end
end

% R' \ (Sb / R) computes inv(R')*Sb*inv(R) without ever forming an inverse.
Minner = (R') \ (Sb / R);
Minner = (Minner + Minner') / 2;          % re-symmetrise after the solves

[Z, Dz] = eig(Minner);
[evals, idx] = sort(diag(Dz), 'descend'); % eig is ascending; we want the best first
Z = Z(:, idx);

% Map back:  w = inv(R) * z, i.e. solve R*w = z.
% rank(S_B) <= c-1, so only the first c-1 eigenvalues are non-zero: there
% are AT MOST c-1 useful Fisherfaces, no matter how big m is.
Wfld = R \ Z(:, 1:(c-1));

Wopt = Vm * Wfld;

% Scale each Fisherface to unit norm.  The Fisher criterion is invariant to
% the scale of each direction, but nearest-neighbour distances are NOT, so we
% must fix a convention; unit norm is the usual one.
for i = 1:size(Wopt, 2)
    nrm = norm(Wopt(:, i));
    if nrm > 1e-10
        Wopt(:, i) = Wopt(:, i) / nrm;
    end
end

info.evals = evals;
info.numFisher = size(Wopt, 2);

info.rankSw = NaN;
info.condSw = NaN;
info.crossCheck = NaN;

if verbose >= 1
    % cond(S_W) is the headline number for the "zero slack" argument: at
    % m = N-c the smallest eigenvalues of S_W are pure estimation noise.
    info.rankSw = rank(Sw);
    info.condSw = cond(Sw);
end
if verbose >= 2
    % one-line cross-check against MATLAB's packaged generalised solver.
    % It should agree to machine precision; if it ever does not, our
    % whitening algebra is wrong.
    Wchk = eig(Sb, Sw, 'chol');
    info.crossCheck = norm(sort(real(Wchk), 'descend') - sort(evals, 'descend')) ...
                      / norm(evals);
end
end


function [rate, cnd] = ldaVariant(V, X, Xte, ytr, yte, c, m, how, epsRel)
% LDAVARIANT  Run LDA in an m-dimensional PCA subspace, coping with a possibly
%             singular S_W in one of three ways.  Used only by section S13.
%
%   how = 'chol'  plain LDA via Cholesky whitening; returns rate = NaN if chol
%                 fails, which is exactly what happens once m > N-c
%         'ridge' replace S_W by S_W + epsRel*trace(S_W)/m * I first
%         'pinv'  solve pinv(S_W)*S_B as an ordinary (non-symmetric) eigenproblem

Vm = V(:, 1:m);
Al = Vm' * X;                      % eigen-coefficients
ab = mean(Al, 2);

Sw = zeros(m, m);
Sb = zeros(m, m);
for j = 1:c
    Aj  = Al(:, ytr == j);
    nj  = size(Aj, 2);
    muj = mean(Aj, 2);
    Dj  = Aj - muj;
    Sw = Sw + Dj * Dj';
    Sb = Sb + nj * (muj - ab) * (muj - ab)';
end
Sw = (Sw + Sw') / 2;
Sb = (Sb + Sb') / 2;
cnd  = cond(Sw);
rate = NaN;

switch how
    case 'chol'
        [R, pp] = chol(Sw);
        if pp ~= 0
            return;                % S_W not positive definite: report failure
        end
        M = (R') \ (Sb / R); M = (M + M') / 2;
        [Z, Dz] = eig(M);
        [~, iz] = sort(diag(Dz), 'descend');
        Wf = Vm * (R \ Z(:, iz(1:(c-1))));
    case 'ridge'
        Sw2 = Sw + epsRel * trace(Sw) / m * eye(m);
        R = chol(Sw2);
        M = (R') \ (Sb / R); M = (M + M') / 2;
        [Z, Dz] = eig(M);
        [~, iz] = sort(diag(Dz), 'descend');
        Wf = Vm * (R \ Z(:, iz(1:(c-1))));
    case 'pinv'
        % pinv(Sw)*Sb is NOT symmetric, so eig can return complex values;
        % we take the real parts, which is itself part of the argument
        % against this route.
        M = pinv(Sw) * Sb;
        [Z, Dz] = eig(M);
        [~, iz] = sort(real(diag(Dz)), 'descend');
        Wf = Vm * real(Z(:, iz(1:(c-1))));
end

for l = 1:size(Wf, 2)
    nrm = norm(Wf(:, l));
    if nrm > 1e-10
        Wf(:, l) = Wf(:, l) / nrm;
    end
end
rate = nnClassify(Wf' * X, ytr, Wf' * Xte, yte);
end


function [centroid, hiShare] = specStats(v, H, W, radGrid)
% SPECSTATS  Spatial-frequency summary of one basis image.
%   centroid  power-weighted mean radial frequency (0 = flat, larger = finer)
%   hiShare   fraction of power above a radial frequency of 0.15
P = abs(fft2(reshape(v, H, W))).^2;
tot = sum(P(:));
centroid = sum(sum(P .* radGrid)) / tot;
hiShare  = sum(sum(P .* (radGrid > 0.15))) / tot;
end


function [Wf, tsec, imagMax] = solveRoute(r, Sw, Sb, Vm, c)
% SOLVEROUTE  One of six ways to solve S_B w = lambda S_W w.  Used by S15.
%   Routes 1, 2, 5, 6 can return complex eigenvalues because the matrix they
%   hand to eig is not symmetric (or, for 5/6, because the pencil is solved
%   in general form); we record how large the imaginary parts get.
t0 = tic;
imagMax = 0;
switch r
    case 1                                   % inv(Sw)*Sb : the naive route
        M = inv(Sw) * Sb;                                        %#ok<MINV>
        [Z, Dz] = eig(M); lam = diag(Dz); imagMax = max(abs(imag(lam)));
        [~, iz] = sort(real(lam), 'descend'); Wf = Vm * real(Z(:, iz(1:(c-1))));
    case 2                                   % same, but solve instead of invert
        M = Sw \ Sb;
        [Z, Dz] = eig(M); lam = diag(Dz); imagMax = max(abs(imag(lam)));
        [~, iz] = sort(real(lam), 'descend'); Wf = Vm * real(Z(:, iz(1:(c-1))));
    case 3                                   % Cholesky whitening  (ours)
        R = chol(Sw); M = (R') \ (Sb / R); M = (M + M') / 2;
        [Z, Dz] = eig(M);
        [~, iz] = sort(diag(Dz), 'descend'); Wf = Vm * (R \ Z(:, iz(1:(c-1))));
    case 4                                   % symmetric square root
        [Us, Ds] = eig(Sw); ds = diag(Ds);
        Sih = Us * diag(1 ./ sqrt(max(ds, eps))) * Us';
        M = Sih * Sb * Sih; M = (M + M') / 2;
        [Z, Dz] = eig(M);
        [~, iz] = sort(diag(Dz), 'descend'); Wf = Vm * (Sih * Z(:, iz(1:(c-1))));
    case 5                                   % MATLAB's general pencil solver (QZ)
        [Z, Dz] = eig(Sb, Sw); lam = diag(Dz); imagMax = max(abs(imag(lam)));
        [~, iz] = sort(real(lam), 'descend'); Wf = Vm * real(Z(:, iz(1:(c-1))));
    case 6                                   % MATLAB's packaged Cholesky route
        [Z, Dz] = eig(Sb, Sw, 'chol'); lam = diag(Dz); imagMax = max(abs(imag(lam)));
        [~, iz] = sort(real(lam), 'descend'); Wf = Vm * real(Z(:, iz(1:(c-1))));
end
tsec = toc(t0);
for l = 1:size(Wf, 2)
    nrm = norm(Wf(:, l));
    if nrm > 1e-10
        Wf(:, l) = Wf(:, l) / nrm;
    end
end
end
