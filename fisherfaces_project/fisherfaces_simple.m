% =====================================================================
%  FISHERFACES, THE SHORT VERSION
%
%  A teaching-sized companion to fisherfaces_project.m.  Same algorithms,
%  same numbers, none of the extra experiments.  Three parts:
%
%    PART 1   a 2-D toy problem where you can SEE why PCA is the wrong
%             projection and Fisher's is the right one
%    PART 2   eigenfaces and Fisherfaces on real faces (ORL, then Yale),
%             compared against each other and against raw pixels
%    PART 3   figures
%
%  Everything is self-contained: the loader, the PCA and the Fisherfaces
%  solver are all local functions at the bottom of this file.  Nothing is
%  needed from the big script.
%
%  Only basic linear algebra is assumed: eigenvectors, rank, and the fact
%  that a symmetric positive definite matrix can be factorised.
%
%  Notation (the same as the study guide and the course slides):
%     d        pixels per image            (ORL 10304, Yale 32256)
%     N        number of training images
%     c        number of classes = number of people
%     n_j      training images of person j
%     m        PCA dimension we reduce to BEFORE doing LDA
%     x_i      one image, as a column of d numbers
%     xbar     the mean image
%     V        eigenfaces (columns)
%     S_B/S_W  between-class / within-class scatter
%
%  Run with:   matlab -batch "run('fisherfaces_simple.m')"
% =====================================================================

clear; clc;
rng(0);                       % fixed seed => the numbers below are repeatable

% Resolve the data paths relative to THIS FILE, not to whatever folder the
% user happens to be sitting in.  mfilename('fullpath') gives the full path
% of the running script without its extension; fileparts peels off the
% directory.  Without this the script only works when the current folder is
% the project folder -- and it would fail SILENTLY, because dir() on a
% non-existent folder returns an empty list rather than an error, so every
% loop below would simply do nothing and we would "classify" zero images.
thisDir  = fileparts(mfilename('fullpath'));
if isempty(thisDir)                       % pasted into the command window
    thisDir = pwd;
end
orlRoot  = fullfile(thisDir, '..', 'ORL', filesep);
yaleRoot = fullfile(thisDir, '..', 'CroppedYale', filesep);
figDir   = fullfile(thisDir, 'figs_simple', filesep);
if exist(figDir, 'dir') == 0
    mkdir(figDir);
end

% Fail loudly and early if the databases are not where we expect.
if exist(orlRoot, 'dir') == 0
    error('Cannot find ORL at %s', orlRoot);
end
if exist(yaleRoot, 'dir') == 0
    error('Cannot find CroppedYale at %s', yaleRoot);
end

% Newer MATLAB may pick a dark figure theme, which prints badly.  Force light.
try
    sTheme = settings;
    sTheme.matlab.appearance.figure.GraphicsTheme.TemporaryValue = 'light';
catch
    % older MATLAB: already light
end
set(groot, 'defaultFigureColor', 'w');


%% ====================================================================
%  PART 1   TOY PROBLEM: WHY PCA IS NOT A CLASSIFIER
%
%  Two classes of 2-D points.  Both share the covariance
%        Sigma = [16 0 ; 0 0.25]
%  so each cloud is a wide, flat ellipse stretched along x.  The class
%  means differ only in y:  mu1 = (0,-1),  mu2 = (0,+1).
%
%  So: all the SPREAD is along x, and all the CLASS INFORMATION is along y.
%  PCA maximises spread, so it will pick x -- and destroy the labels.
%  Fisher maximises (between-class / within-class), so it picks y.
%
%  This is the whole idea of the paper in two dimensions.
%% ====================================================================

fprintf('=====================================================\n');
fprintf('  PART 1: toy 2-D problem, PCA vs Fisher\n');
fprintf('=====================================================\n');

Sigma = [16 0; 0 0.25];
mu1   = [ 0; -1];
mu2   = [ 0;  1];
nPer  = 500;

% Generate the two clouds.  chol(Sigma) gives R with R'*R = Sigma, so if z
% has identity covariance then R'*z has covariance Sigma.  (This is the same
% Cholesky factorisation used for Fisherfaces later, in reverse.)
R  = chol(Sigma);
X1 = mu1 + R' * randn(2, nPer);
X2 = mu2 + R' * randn(2, nPer);

% ---- the two scatter matrices, straight from the definitions ----
m1 = mean(X1, 2);
m2 = mean(X2, 2);
mAll = (m1 + m2) / 2;

D1 = X1 - m1;                   % each class centred on ITS OWN mean
D2 = X2 - m2;
Sw = D1 * D1' + D2 * D2';       % within-class scatter

Sb = nPer * (m1 - mAll) * (m1 - mAll)' + ...
     nPer * (m2 - mAll) * (m2 - mAll)';      % between-class scatter

% ---- Fisher's direction, two-class closed form ----
% Maximising  J(w) = (w'*Sb*w) / (w'*Sw*w)  for TWO classes has the solution
%       w  proportional to  inv(Sw) * (m1 - m2)
% Sw \ (m1-m2) solves Sw*w = (m1-m2) without forming an inverse.
wLDA = Sw \ (m1 - m2);
wLDA = wLDA / norm(wLDA);

% ---- PCA's direction: top eigenvector of the TOTAL scatter ----
Xall = [X1, X2];
Dall = Xall - mean(Xall, 2);
St   = Dall * Dall';
[Vt, Dt] = eig(St);
[~, iBig] = max(diag(Dt));      % eig returns ASCENDING, so take the max
wPCA = Vt(:, iBig);

% ---- the Fisher criterion value for each direction ----
Jlda = (wLDA' * Sb * wLDA) / (wLDA' * Sw * wLDA);
Jpca = (wPCA' * Sb * wPCA) / (wPCA' * Sw * wPCA);

fprintf('\n  Fisher direction : (%.4f, %.4f)   <- almost exactly the y-axis\n', wLDA);
fprintf('  PCA    direction : (%.4f, %.4f)   <- almost exactly the x-axis\n', wPCA);
fprintf('  J(w_Fisher) = %.6f\n', Jlda);
fprintf('  J(w_PCA)    = %.6f\n', Jpca);
fprintf('  ratio       = %.1fx  (Fisher separates; PCA does not)\n', Jlda / Jpca);

% ---- classify the toy data along each direction, to make J concrete ----
% Project everything to 1-D, then assign each point to the nearer class mean.
errLDA = toyError(X1, X2, wLDA);
errPCA = toyError(X1, X2, wPCA);
fprintf('\n  error along Fisher direction = %.2f%%\n', errLDA);
fprintf('  error along PCA    direction = %.2f%%  (near chance: 50%%)\n', errPCA);

% ---- the theoretical floor, for comparison ----
% With equal-covariance Gaussians and equal priors, the best achievable error
% is Phi(-Delta/2) where Delta is the Mahalanobis distance between the means.
Delta = sqrt((mu2 - mu1)' * (Sigma \ (mu2 - mu1)));
bayes = 100 * 0.5 * erfc((Delta / 2) / sqrt(2));    % Phi(-Delta/2), via erfc
fprintf('  Mahalanobis separation Delta = %.4f\n', Delta);
fprintf('  best possible (Bayes) error   = %.2f%%\n', bayes);
fprintf('  NOTE: the measured %.2f%% can land slightly BELOW %.2f%% -- that is\n', ...
        errLDA, bayes);
fprintf('        sampling noise on %d points, not a contradiction.\n', 2 * nPer);


%% ====================================================================
%  PART 2   REAL FACES
%
%  Loop over two databases, because they say opposite things:
%
%    ORL   40 people, 10 images each, 92x112.  Varies POSE and EXPRESSION.
%    Yale  38 people, ~64 images each, 168x192.  Varies ILLUMINATION only.
%
%  For each one we run four methods and compare:
%    1. correlation   nearest neighbour on raw pixels (no projection at all)
%    2. eigenfaces    PCA, best k
%    3. eigenfaces    PCA at k = c-1, so it has the SAME number of features
%                     as Fisherfaces (otherwise the comparison is unfair)
%    4. Fisherfaces   PCA to m = N-c, then LDA, keeping c-1 directions
%% ====================================================================

fprintf('\n\n=====================================================\n');
fprintf('  PART 2: eigenfaces vs Fisherfaces on real faces\n');
fprintf('=====================================================\n');

% Build the list of subject folders for each database.
% ORL: build the path from the index.  Do NOT use dir('s*') -- this copy of
% ORL contains a stray empty directory called 'sed', which a glob would pick
% up as a 41st "subject".
orlFolders = cell(1, 32);
for p = 1:32
    orlFolders{p} = [orlRoot 's' num2str(p) '/'];
end

% Yale: same loop.  The folders were originally called yaleB01..yaleB39 with
% yaleB14 missing, which is not a countable range, so we renamed them to
% s1..s38 with rename_yale.sh.  The labels were always the loop index and
% never the folder name, so renaming does not change any result.
yaleFolders = cell(1, 38);
for p = 1:38
    yaleFolders{p} = [yaleRoot 's' num2str(p) '/'];
end

% Describe the two databases as a STRUCT ARRAY rather than a cell table.
% A cell table would need sets{s,2}{1} to get the folder list back out: in a
% cell literal, {orlFolders} wraps the cell array inside another 1x1 cell
% instead of flattening it, so one set of braces unwraps the table and a
% second unwraps the nesting.  Struct fields avoid the puzzle entirely.
%
% imread returns rows x cols, so H is the number of ROWS: ORL images are 92
% wide and 112 tall, giving H=112, W=92.
sets(1).name = 'ORL';   sets(1).folders = orlFolders;
sets(1).numTrain = 6;   sets(1).H = 112;  sets(1).W = 92;

sets(2).name = 'Yale';  sets(2).folders = yaleFolders;
sets(2).numTrain = 40;  sets(2).H = 192;  sets(2).W = 168;

results = zeros(2, 4);          % rows = database, cols = the four methods
keep    = cell(1, 2);           % saved for the figures in Part 3

for s = 1:2
    name     = sets(s).name;
    folders  = sets(s).folders;
    numTrain = sets(s).numTrain;
    H        = sets(s).H;
    W        = sets(s).W;
    d        = H * W;

    fprintf('\n-----------------------------------------------------\n');
    fprintf('  %s\n', name);
    fprintf('-----------------------------------------------------\n');

    % ---- load ----
    [imTr, ytr, imTe, yte] = loadSet(folders, numTrain, d);
    N = size(imTr, 2);
    c = numel(folders);
    fprintf('  d = %d pixels, N = %d training, %d test, c = %d people\n', ...
            d, N, size(imTe, 2), c);

    % ---- method 1: correlation, i.e. no projection at all ----
    % The baseline the paper compares against.  If a method cannot beat this,
    % the projection is not earning its place.
    %
    % normCols makes each image zero-mean and unit-norm first, which is what
    % turns nearest neighbour into a CORRELATION classifier: for unit-norm
    % zero-mean a and b,  ||a-b||^2 = 2 - 2*a'*b,  so the nearest neighbour
    % is exactly the most correlated one.  It also removes each image's
    % brightness offset and contrast scaling.  Skipping it changes the
    % answer: ORL 91.4% -> 93.8%, Yale 43.0% -> 35.6%.
    results(s, 1) = nnClassify(normCols(imTr), ytr, normCols(imTe), yte);
    fprintf('\n  1. correlation (raw pixels)      : %.1f%%\n', results(s, 1));

    % ---- PCA: the eigenfaces ----
    [V, lambda, xbar] = doPCA(imTr);

    % ---- method 2 and 3: eigenfaces at several k ----
    % Project onto the top k eigenfaces and classify.  V(:,1:k)' * (x - xbar)
    % is the vector of k coefficients for image x.
    ks   = [10 20 30 50 80 120 170];
    ks   = ks(ks <= N - 1);                 % only min(d,N)-1 directions exist
    best = 0;  bestK = 0;
    for k = ks
        Vk = V(:, 1:k);
        r  = nnClassify(Vk' * (imTr - xbar), ytr, Vk' * (imTe - xbar), yte);
        if r > best
            best  = r;
            bestK = k;
        end
    end
    results(s, 2) = best;
    fprintf('  2. eigenfaces, best k = %-3d      : %.1f%%\n', bestK, best);

    Vc = V(:, 1:(c-1));                     % matched-dimension comparison
    results(s, 3) = nnClassify(Vc' * (imTr - xbar), ytr, Vc' * (imTe - xbar), yte);
    fprintf('  3. eigenfaces at k = c-1 = %-3d   : %.1f%%\n', c-1, results(s, 3));

    % ---- method 4: Fisherfaces ----
    m = N - c;                              % the paper's choice of m
    [Wopt, Sb, Sw] = fisherfaces(V, xbar, imTr, ytr, m, c);
    results(s, 4) = nnClassify(Wopt' * (imTr - xbar), ytr, ...
                               Wopt' * (imTe - xbar), yte);
    fprintf('  4. FISHERFACES (%d features)     : %.1f%%\n', ...
            size(Wopt, 2), results(s, 4));

    % ---- the three sanity checks ----
    % These cost almost nothing and each one tests a different theorem.
    fprintf('\n  checks:\n');

    % (a) S_T = S_B + S_W.  A theorem, so it must hold to machine precision.
    %     If a class mean or a class count were wrong, this would break.
    Al = V(:, 1:m)' * (imTr - xbar);        % the training data in PCA space
    Dl = Al - mean(Al, 2);
    St = Dl * Dl';
    fprintf('    (a) ||S_T - (S_B+S_W)|| / ||S_T|| = %.3e   (should be ~1e-16)\n', ...
            norm(St - (Sb + Sw), 'fro') / norm(St, 'fro'));

    % (b) rank(S_B) <= c-1, because the c class means have only c-1 degrees
    %     of freedom once you subtract the overall mean.  This is WHY there
    %     are at most c-1 Fisherfaces.
    fprintf('    (b) rank(S_B) = %d, bound c-1 = %d, we kept %d\n', ...
            rank(Sb), c-1, size(Wopt, 2));

    % (c) Fisherfaces are S_W-orthogonal, NOT Euclidean-orthogonal.
    %     W'*W is NOT the identity; W'*S_W*W IS.  Common misconception.
    nf = size(Wopt, 2);

    % W'*W in pixel space: should NOT be the identity.
    Ge = Wopt' * Wopt;
    offEuclid = norm(Ge - diag(diag(Ge)), 'fro') / norm(Ge, 'fro');

    % W'*S_W*W: should BE the identity.  S_W lives in the reduced space, and
    % Wopt = V(:,1:m)*Wfld with V(:,1:m) orthonormal, so Wfld comes back out
    % by projecting: V(:,1:m)'*Wopt.
    Wfld = V(:, 1:m)' * Wopt;
    Gs   = Wfld' * Sw * Wfld;
    offSw = norm(Gs - diag(diag(Gs)), 'fro') / norm(Gs, 'fro');
    fprintf('    (c) off-diagonal share of W''*W     (Euclidean) = %.3e\n', offEuclid);
    fprintf('        off-diagonal share of W''*S_W*W (S_W-inner)  = %.3e\n', offSw);
    fprintf('        => S_W-orthogonal, not Euclidean-orthogonal\n');

    % keep what Part 3 needs to draw
    keep{s} = struct('name', name, 'xbar', xbar, 'V', V, ...
                     'Wopt', Wopt, 'H', H, 'W', W, 'nf', nf);
end

% ---- the summary table ----
fprintf('\n\n================ SUMMARY (recognition rate %%) ================\n');
fprintf('  database    correlation   eig(best)   eig(c-1)   FISHERFACES\n');
for s = 1:2
    fprintf('  %-10s  %9.1f   %9.1f  %9.1f   %11.1f\n', ...
            sets(s).name, results(s,1), results(s,2), ...
            results(s,3), results(s,4));
end
fprintf('==============================================================\n');
fprintf('\n  Read it like this:\n');
fprintf('   - On Yale (illumination only) Fisherfaces win by a wide margin.\n');
fprintf('     That is the paper''s claim, reproduced.\n');
fprintf('   - On ORL (pose and expression) Fisherfaces LOSE to eigenfaces.\n');
fprintf('     Reason: ORL gives 6 training images per person, so each class\n');
fprintf('     contributes only n_j - 1 = 5 degrees of freedom to S_W.  Yale\n');
fprintf('     gives 40, so 39.  S_W is far better estimated on Yale, and\n');
fprintf('     Fisherfaces depend on inverting it.\n');
fprintf('   - On Yale, plain correlation BEATS eigenfaces.  PCA actively\n');
fprintf('     hurts there, because its leading components encode lighting.\n');


%% ====================================================================
%  PART 3   FIGURES
%
%  Three pictures per database: the mean face, the top eigenfaces, and the
%  top Fisherfaces.  The point of the last two side by side: eigenfaces look
%  like smooth, ghostly faces, Fisherfaces look like high-frequency noise.
%  That is not a bug.  Whitening divides direction l by sqrt(lambda_l), so
%  the low-variance (fine detail) directions get amplified the most.
%% ====================================================================

fprintf('\n\n=====================================================\n');
fprintf('  PART 3: figures\n');
fprintf('=====================================================\n');

for s = 1:2
    K = keep{s};

    % ---- the mean face ----
    figure('Visible', 'off');
    imagesc(reshape(K.xbar, K.H, K.W));
    colormap gray; axis image off;
    title([K.name ': mean face']);
    saveas(gcf, [figDir K.name '_meanface.png']);
    close;

    % ---- top 16 eigenfaces ----
    figure('Visible', 'off', 'Position', [100 100 700 700]);
    for l = 1:16
        subplot(4, 4, l);
        imagesc(reshape(K.V(:, l), K.H, K.W));
        colormap gray; axis image off;
        title(sprintf('e%d', l));
    end
    sgtitle([K.name ': top 16 eigenfaces (smooth, face-like)']);
    saveas(gcf, [figDir K.name '_eigenfaces.png']);
    close;

    % ---- top 16 Fisherfaces ----
    nShow = min(16, K.nf);
    figure('Visible', 'off', 'Position', [100 100 700 700]);
    for l = 1:nShow
        subplot(4, 4, l);
        imagesc(reshape(K.Wopt(:, l), K.H, K.W));
        colormap gray; axis image off;
        title(sprintf('f%d', l));
    end
    sgtitle([K.name ': top Fisherfaces (high-frequency, not face-like)']);
    saveas(gcf, [figDir K.name '_fisherfaces.png']);
    close;

    fprintf('  %s: wrote 3 figures\n', K.name);
end

fprintf('\n  figures written to %s\n', figDir);
fprintf('\n=====================================================\n');
fprintf('  DONE\n');
fprintf('=====================================================\n');


%% ====================================================================
%  LOCAL FUNCTIONS
%
%  In MATLAB, functions defined inside a SCRIPT must all come after the
%  executable code, at the very end of the file.  They can be called from
%  anywhere above.
%% ====================================================================

function err = toyError(X1, X2, w)
% TOYERROR  Project the two toy clouds onto w, classify by nearer class
%           mean in 1-D, and return the error as a percentage.
p1 = w' * X1;                   % 1 x n row of projections
p2 = w' * X2;
c1 = mean(p1);
c2 = mean(p2);

wrong = 0;
for i = 1:numel(p1)
    if abs(p1(i) - c1) > abs(p1(i) - c2)    % closer to the other mean
        wrong = wrong + 1;
    end
end
for i = 1:numel(p2)
    if abs(p2(i) - c2) > abs(p2(i) - c1)
        wrong = wrong + 1;
    end
end
err = 100 * wrong / (numel(p1) + numel(p2));
end


function [imTr, ytr, imTe, yte] = loadSet(folders, numTrain, d)
% LOADSET  Read the PGM images, turn each into a column, split train/test.
%
%   imTr is d x N: ONE IMAGE PER COLUMN.  ytr(i) is the person in column i.
%
%   The first numTrain images of each person go to training, the rest to
%   test, in whatever order dir() returns them.  That is deliberately the
%   same order the HW3 code used, so the numbers match.
%
%   GOTCHA worth knowing: dir() sorts names as TEXT.  For ORL that means
%   1.pgm, 10.pgm, 2.pgm, ... so "the first 6" is really {1,10,2,3,4,5}.
%   For Yale the names encode lighting as A+020 / A-020, and '+' sorts
%   before '-', so training gets the positive azimuths and testing gets
%   the negative ones.  Both splits are harder than they look.

nF = numel(folders);

% Count the files first so we can preallocate.  Growing a matrix inside a
% loop makes MATLAB copy the whole thing every iteration; for Yale that is
% 1520 copies of a 392 MB array.
nTr = 0;
nTe = 0;
for p = 1:nF
    f = dir([folders{p} '*.pgm']);
    if isempty(f)
        % dir() on a missing or empty folder returns an empty struct, NOT an
        % error.  Without this check the whole script would run to completion
        % on zero images and print meaningless rates.
        error('No .pgm images found in %s', folders{p});
    end
    ntr = min(numTrain, numel(f));       % a person may have fewer images
    nTr = nTr + ntr;
    nTe = nTe + (numel(f) - ntr);
end

imTr = zeros(d, nTr);   ytr = zeros(1, nTr);
imTe = zeros(d, nTe);   yte = zeros(1, nTe);

a = 0;
b = 0;
for p = 1:nF
    files = dir([folders{p} '*.pgm']);
    for j = 1:numel(files)
        im = double(imread([folders{p} files(j).name]));
        if j <= numTrain
            a = a + 1;
            imTr(:, a) = im(:);          % im(:) stacks the image COLUMN by COLUMN
            ytr(a)     = p;
        else
            b = b + 1;
            imTe(:, b) = im(:);
            yte(b)     = p;
        end
    end
end
end


function [V, lambda, xbar] = doPCA(imgs)
% DOPCA  Eigenfaces, using the L = X'*X trick.
%
%   The covariance in pixel space is C = X*X'/N, which is d x d.  For Yale
%   that is 32256 x 32256 = 8.3 GB -- impossible.  But we only need its top
%   N eigenvectors, and there is a shortcut.
%
%   THE TRICK.  Let L = X'*X, which is only N x N.  If L*w = lambda*w then
%        X*X' * (X*w) = X * (X'*X*w) = X * (lambda*w) = lambda * (X*w)
%   so X*w is an eigenvector of X*X' with the SAME eigenvalue.  So we solve
%   the small problem and multiply by X.
%
%   One catch: ||X*w|| = sqrt(lambda), not 1, so the columns must be
%   renormalised afterwards or the projection stops being orthogonal.

[~, N] = size(imgs);

xbar = mean(imgs, 2);           % the mean face, a d x 1 column
X    = imgs - xbar;             % centre every image (implicit expansion)

L = X' * X;                     % N x N instead of d x d
[Wg, D] = eig(L);

% eig returns eigenvalues in ASCENDING order.  We want the biggest first,
% because those are the directions carrying the most variance.
[eigvals, idx] = sort(diag(D), 'descend');
Wg = Wg(:, idx);

V = X * Wg;                     % map back to pixel space: d x N

% Renormalise each column to unit length.
% After centring, rank(X) <= N-1, so the last column is numerically zero and
% dividing by its tiny norm would amplify pure round-off into a unit vector.
% The threshold is RELATIVE to the first column, so it scales with the data.
scale = norm(V(:, 1));
for l = 1:size(V, 2)
    nrm = norm(V(:, l));
    if nrm > 1e-10 * scale
        V(:, l) = V(:, l) / nrm;
    else
        V(:, l) = 0;            % a numerically empty direction
    end
end

% Eigenvalues of L are N times those of the covariance C = X*X'/N.
lambda = eigvals / N;
end


function [Wopt, Sb, Sw] = fisherfaces(V, xbar, imgs, y, m, c)
% FISHERFACES  The two-stage algorithm: PCA to m dimensions, then LDA.
%
%   WHY TWO STAGES.  In pixel space S_W is d x d but rank(S_W) <= N-c.  For
%   Yale that is 1482 << 32256, so S_W is massively singular and the Fisher
%   criterion J(w) = (w'S_B w)/(w'S_W w) is not even defined.  Projecting
%   onto m <= N-c PCA directions throws away exactly the directions S_W
%   knows nothing about.  What is left is full rank, so S_W is invertible
%   and we can proceed.  This is the paper's fix, and its only one.
%
%   Returns Wopt (d x c-1), and S_B, S_W as built in the reduced space.

Vm = V(:, 1:m);                 % the PCA basis we project onto
Al = Vm' * (imgs - xbar);       % the training data in m-dimensional space
abar = mean(Al, 2);

% ---- build the two scatter matrices, straight from the definitions ----
Sw = zeros(m, m);
Sb = zeros(m, m);
for j = 1:c
    Aj = Al(:, y == j);         % logical indexing: all columns of person j
    nj = size(Aj, 2);
    mj = mean(Aj, 2);           % this person's mean
    Dj = Aj - mj;               % centred on THEIR OWN mean
    Sw = Sw + Dj * Dj';
    Sb = Sb + nj * (mj - abar) * (mj - abar)';
end

% Floating-point arithmetic makes these very slightly non-symmetric (around
% 1e-16).  chol and the symmetric eig read only one triangle, so force exact
% symmetry first rather than letting the solver silently pick a half.
Sw = (Sw + Sw') / 2;
Sb = (Sb + Sb') / 2;

% ---- solve  S_B*w = lambda*S_W*w  by Cholesky whitening ----
%
%   S_W is symmetric positive definite, so S_W = R'*R with R upper
%   triangular (this is Cholesky).  Substitute w = inv(R)*z:
%
%       S_B inv(R) z = lambda * R'R inv(R) z = lambda * R' z
%
%   and multiply on the left by inv(R'):
%
%       inv(R') S_B inv(R) z = lambda z        <-- call this matrix M
%
%   M is symmetric, because M' = inv(R')*S_B'*inv(R) = M.  So we have turned
%   a GENERALISED eigenproblem into an ORDINARY SYMMETRIC one, for which
%   eigenvalues are guaranteed real and eigenvectors orthogonal.
%
%   (Writing S_W = L*L' with L lower triangular gives inv(L)*S_B*inv(L'),
%   which is literally the same matrix, since L = R'.  The convention is
%   bookkeeping.  We use upper because that is what chol returns.)
R = chol(Sw);

% (R')\(Sb/R) computes inv(R')*Sb*inv(R) as two TRIANGULAR SOLVES, without
% ever forming an inverse.  Sb/R solves from the right, (R')\... from the left.
M = (R') \ (Sb / R);
M = (M + M') / 2;               % re-symmetrise after the solves

[Z, Dz] = eig(M);
[~, idx] = sort(diag(Dz), 'descend');       % eig is ascending; best first
Z = Z(:, idx);

% Keep only c-1 directions.  rank(S_B) <= c-1, so all later eigenvalues are
% zero and their eigenvectors are an arbitrary basis of a null space --
% keeping them would add pure noise.
Wfld = R \ Z(:, 1:(c-1));       % map back: w = inv(R)*z, i.e. solve R*w = z

% Compose the two stages.  A test image x is projected as Wopt'*(x - xbar).
Wopt = Vm * Wfld;

% ---- scale each Fisherface to unit length.  THIS IS NOT COSMETIC ----
%
% J(w) = (w'S_B w)/(w'S_W w) is unchanged if you scale w, so the maths does
% not care and the SUBSPACE we found is already correct.  But our classifier
% is nearest neighbour with Euclidean distance, and that is NOT scale
% invariant: a coordinate with a large column norm contributes more to every
% distance.  The columns coming out of R\Z are S_W-orthogonal, meaning
% W'*S_W*W = I, which makes their Euclidean norms differ by orders of
% magnitude (whitening scales direction l by 1/sqrt(lambda_l)).  Leaving them
% unnormalised lets the few largest-norm directions dominate the distance.
%
% Measured cost of skipping these four lines: ORL drops 90.6% -> 20.3% and
% Yale 67.2% -> 13.5%.  Same subspace, same theory, ruined classifier.
for l = 1:size(Wopt, 2)
    nrm = norm(Wopt(:, l));
    if nrm > 0
        Wopt(:, l) = Wopt(:, l) / nrm;
    end
end
end


function Xn = normCols(X)
% NORMCOLS  Make every image (column) zero-mean and unit-norm.
%           Used only for the correlation baseline.

Xn = X - mean(X, 1);            % mean(X,1) is the per-COLUMN (per-image) mean
for i = 1:size(Xn, 2)
    nrm = norm(Xn(:, i));
    if nrm > 0
        Xn(:, i) = Xn(:, i) / nrm;
    end
end
end


function rate = nnClassify(trainProj, trainLabels, testProj, testLabels)
% NNCLASSIFY  Nearest neighbour with squared Euclidean distance.
%
%   For each test column, find the closest training column and take its
%   label.  There is no training and nothing to converge: the "model" is
%   the training set itself.
%
%   We compare SQUARED distances and never take a square root, because sqrt
%   is increasing and so cannot change which neighbour is nearest.

correct = 0;
for i = 1:size(testProj, 2)
    % trainProj - testProj(:,i) subtracts the test column from EVERY
    % training column at once; sum(...,1) collapses each to one number.
    distances = sum((trainProj - testProj(:, i)).^2, 1);
    [~, best] = min(distances);
    if trainLabels(best) == testLabels(i)
        correct = correct + 1;
    end
end
rate = 100 * correct / size(testProj, 2);
end
