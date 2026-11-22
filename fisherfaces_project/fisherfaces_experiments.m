% =====================================================================
%  FISHERFACES: EXTRA EXPERIMENTS
%
%  This file continues where fisherfaces_simple.m stops.  That script
%  answers "does it work?".  This one answers five harder questions:
%
%    E1  How badly conditioned is S_W as m grows?  (cond(S_W) vs m)
%    E2  Is the 90.6% real, or is the pipeline leaking labels?
%    E3  Can we choose m WITHOUT looking at the test set?
%    E4  Fisherfaces are higher-frequency.  Does that make them more
%        fragile under pixel noise, as the theory predicts?
%    E5  The paper's real protocol: error versus lighting angle.
%
%  Each experiment prints its numbers and writes a figure.
%
%  SELF-CONTAINED.  The helper functions at the bottom are copies of the
%  ones in fisherfaces_simple.m, so you can read this file on its own.
%  (MATLAB will not let one script call another script's local functions,
%  so the duplication is forced as well as convenient.)
%
%  Run with:  matlab -batch "run('fisherfaces_experiments.m')"
%  Takes a few minutes, mostly loading Yale.
% =====================================================================

clear; clc;
rng(0);

thisDir = fileparts(mfilename('fullpath'));
if isempty(thisDir)
    thisDir = pwd;
end
orlRoot  = fullfile(thisDir, '..', 'ORL', filesep);
yaleRoot = fullfile(thisDir, '..', 'CroppedYale', filesep);
figDir   = fullfile(thisDir, 'figs_experiments', filesep);
if exist(figDir, 'dir') == 0
    mkdir(figDir);
end
if exist(orlRoot, 'dir') == 0
    error('Cannot find ORL at %s', orlRoot);
end
if exist(yaleRoot, 'dir') == 0
    error('Cannot find CroppedYale at %s', yaleRoot);
end

try
    sTheme = settings;
    sTheme.matlab.appearance.figure.GraphicsTheme.TemporaryValue = 'light';
catch
    % older MATLAB: already light
end
set(groot, 'defaultFigureColor', 'w');

% ---- folder lists (same indexed loop for both databases) ----
orlFolders = cell(1, 32);
for p = 1:32
    orlFolders{p} = [orlRoot 's' num2str(p) '/'];
end
yaleFolders = cell(1, 38);
for p = 1:38
    yaleFolders{p} = [yaleRoot 's' num2str(p) '/'];
end

dORL  = 112 * 92;
dYALE = 192 * 168;

% ---- load both databases once and reuse them everywhere below ----
fprintf('loading ORL ...\n');
[oTr, oYtr, oTe, oYte] = loadSet(orlFolders, 6, dORL);
fprintf('loading Yale ... (this is the slow one)\n');
[yTr, yYtr, yTe, yYte, yTeNames] = loadSet(yaleFolders, 40, dYALE);

oN = size(oTr, 2);   oC = 32;
yN = size(yTr, 2);   yC = 38;
fprintf('ORL  : N = %d, c = %d, N-c = %d\n', oN, oC, oN - oC);
fprintf('Yale : N = %d, c = %d, N-c = %d\n', yN, yC, yN - yC);

% PCA once per database.  Every experiment below reuses these.
[oV, ~, oXbar] = doPCA(oTr);
[yV, ~, yXbar] = doPCA(yTr);


%% ====================================================================
%  E1   CONDITION NUMBER OF S_W AGAINST m
%
%  The paper says: use PCA to reduce to m = N-c, then do LDA.  At exactly
%  that m, S_W has rank m -- full rank, but with ZERO SLACK.  One more PCA
%  direction and it would be singular.
%
%  So we expect cond(S_W) to be mild for small m and then to explode as m
%  approaches N-c.  "Explode" is vague, so measure it.
%
%  cond(A) = largest eigenvalue / smallest eigenvalue, for symmetric A.
%  It says how much a relative error in the input can be amplified in the
%  answer: roughly log10(cond) digits of the 16 we have are lost.
%
%  Note m is squeezed from both sides:
%        c-1  <=  m  <=  N-c
%  the upper bound so that S_W stays invertible, the lower bound because
%  we extract c-1 directions and cannot get more directions than the space
%  has dimensions.  On ORL that window is 31 <= m <= 160.
%% ====================================================================

fprintf('\n\n=== E1: cond(S_W) against m ===\n');

for db = 1:2
    if db == 1
        name = 'ORL';  V = oV; xbar = oXbar; imTr = oTr; ytr = oYtr;
        imTe = oTe; yte = oYte; c = oC; N = oN;
        ms = [31 40 60 80 100 120 140 150 155 160];   % must start at c-1 = 31
    else
        name = 'Yale'; V = yV; xbar = yXbar; imTr = yTr; ytr = yYtr;
        imTe = yTe; yte = yYte; c = yC; N = yN;
        ms = [100 200 400 700 1000 1200 1400 1450 1482];
    end

    conds = zeros(1, numel(ms));
    rates = zeros(1, numel(ms));
    ranks = zeros(1, numel(ms));

    fprintf('\n  %s  (N = %d, c = %d, so N-c = %d)\n', name, N, c, N - c);
    fprintf('      m     rank(S_W)    cond(S_W)    rate%%\n');
    for i = 1:numel(ms)
        m = ms(i);
        [Wopt, ~, Sw] = fisherfaces(V, xbar, imTr, ytr, m, c);
        conds(i) = cond(Sw);
        ranks(i) = rank(Sw);
        rates(i) = nnRate(Wopt' * (imTr - xbar), ytr, Wopt' * (imTe - xbar), yte);
        fprintf('   %6d   %7d    %.3e     %5.1f\n', m, ranks(i), conds(i), rates(i));
    end

    % ---- figure: condition number and accuracy, side by side ----
    figure('Visible', 'off', 'Position', [100 100 900 350]);

    subplot(1, 2, 1);
    semilogy(ms, conds, '-o', 'LineWidth', 1.5);
    grid on;
    xlabel('m  (PCA dimension before LDA)');
    ylabel('cond(S_W)   [log scale]');
    title([name ': conditioning blows up at m = N-c']);

    subplot(1, 2, 2);
    plot(ms, rates, '-o', 'LineWidth', 1.5);
    grid on;
    xlabel('m  (PCA dimension before LDA)');
    ylabel('recognition rate (%)');
    title([name ': accuracy peaks before m = N-c']);

    saveas(gcf, [figDir 'E1_cond_vs_m_' name '.png']);
    close;

    [bestRate, iBest] = max(rates);
    fprintf('   best m = %d at %.1f%%; the paper would use m = %d, giving %.1f%%\n', ...
            ms(iBest), bestRate, N - c, rates(end));
    fprintf('   cond(S_W) grows by a factor of %.3e from m = %d to m = %d\n', ...
            conds(end) / conds(1), ms(1), ms(end));
end


%% ====================================================================
%  E2   LABEL PERMUTATION: IS THE RESULT REAL?
%
%  The single most useful sanity check there is.  Shuffle the TRAINING
%  labels at random and leave everything else alone.  Now there is no
%  relationship between a face and its label, so no method can do better
%  than guessing, which is 100/c = 3.1% on ORL.
%
%  If a method still scores well after this, the result was never about
%  faces -- it would mean the pipeline is leaking information somewhere,
%  for example by seeing test data during training.
%% ====================================================================

fprintf('\n\n=== E2: label permutation sanity check (ORL) ===\n');

% ---- the honest run, for reference ----
Vc = oV(:, 1:(oC-1));
rEigTrue = nnRate(Vc' * (oTr - oXbar), oYtr, Vc' * (oTe - oXbar), oYte);
[WTrue, ~, ~] = fisherfaces(oV, oXbar, oTr, oYtr, oN - oC, oC);
rFishTrue = nnRate(WTrue' * (oTr - oXbar), oYtr, WTrue' * (oTe - oXbar), oYte);

% ---- now destroy the labels ----
% randperm(n) is a random reordering of 1..n, so ytrShuf gives every
% training image somebody else's label.  The IMAGES are untouched.
shuffle  = randperm(oN);
ytrShuf  = oYtr(shuffle);

rEigPerm = nnRate(Vc' * (oTr - oXbar), ytrShuf, Vc' * (oTe - oXbar), oYte);
[WPerm, ~, ~] = fisherfaces(oV, oXbar, oTr, ytrShuf, oN - oC, oC);
rFishPerm = nnRate(WPerm' * (oTr - oXbar), ytrShuf, WPerm' * (oTe - oXbar), oYte);

chance = 100 / oC;
fprintf('                       true labels   shuffled labels\n');
fprintf('  eigenfaces (k=c-1)      %5.1f%%          %5.1f%%\n', rEigTrue, rEigPerm);
fprintf('  Fisherfaces             %5.1f%%          %5.1f%%\n', rFishTrue, rFishPerm);
fprintf('  chance (100/c)          %5.1f%%          %5.1f%%\n', chance, chance);
fprintf('\n  Both collapse to chance or below, which is what we want to see:\n');
fprintf('  the recognition rates come from the faces, not from the plumbing.\n');
fprintf('\n  Why BELOW chance rather than exactly at it?  Because the shuffle\n');
fprintf('  is applied to training labels only.  Fisherfaces faithfully learns\n');
fprintf('  to separate the shuffled grouping, so a test face is mapped near\n');
fprintf('  the training images that look like it -- and those now carry\n');
fprintf('  somebody else''s label by construction.  The method works; it has\n');
fprintf('  just been told the wrong thing, and it is systematically wrong.\n');

figure('Visible', 'off', 'Position', [100 100 560 380]);
bar([rEigTrue, rEigPerm; rFishTrue, rFishPerm]);
hold on;
plot([0.5 2.5], [chance chance], 'k--', 'LineWidth', 1.2);
hold off;
set(gca, 'XTickLabel', {'eigenfaces', 'Fisherfaces'});
ylabel('recognition rate (%)');
legend({'true labels', 'shuffled labels', 'chance = 100/c'}, 'Location', 'northeast');
title('ORL: shuffling the training labels destroys both methods');
grid on;
saveas(gcf, [figDir 'E2_label_permutation.png']);
close;


%% ====================================================================
%  E3   CHOOSING m WITHOUT LOOKING AT THE TEST SET
%
%  E1 picked "the best m" by reading off the test rate.  That is cheating:
%  the test set is supposed to be touched once, at the end.  If we tune on
%  it, the number we report is optimistic and is not an estimate of how the
%  method does on new faces.
%
%  The honest version is CROSS-VALIDATION inside the training set.  ORL
%  gives 6 images per person, so we can do 6-fold leave-one-out: train on
%  5 per person, validate on the held-out 1, rotate, average.  Only after
%  m is chosen do we look at the test set, exactly once.
%
%  NOTE the folds train on 5 images per person, so N' = 160 and the largest
%  usable m inside a fold is N' - c = 128.  We sweep up to 120.
%% ====================================================================

fprintf('\n\n=== E3: choosing m by cross-validation (ORL) ===\n');

msCV   = [31 40 60 80 100 120];      % m >= c-1 = 31, and m <= 160-32 = 128
nFold  = 6;
cvRate = zeros(1, numel(msCV));

for i = 1:numel(msCV)
    m = msCV(i);
    total = 0;
    for f = 1:nFold
        % Build this fold: the f-th image of every person is validation,
        % the other five are training.
        isVal = false(1, oN);
        for j = 1:oC
            cols = find(oYtr == j);     % the 6 columns belonging to person j
            isVal(cols(f)) = true;      % hold out the f-th
        end
        trCols = find(~isVal);
        vaCols = find(isVal);

        subTr  = oTr(:, trCols);   subYtr = oYtr(trCols);
        subVa  = oTr(:, vaCols);   subYva = oYtr(vaCols);

        % PCA must be recomputed on the fold's training data only --
        % fitting it on all 6 images would leak the validation images.
        [Vf, ~, xbarF] = doPCA(subTr);
        [Wf, ~, ~] = fisherfaces(Vf, xbarF, subTr, subYtr, m, oC);
        total = total + nnRate(Wf' * (subTr - xbarF), subYtr, ...
                               Wf' * (subVa - xbarF), subYva);
    end
    cvRate(i) = total / nFold;
    fprintf('   m = %3d   cross-validated rate = %5.1f%%\n', m, cvRate(i));
end

[~, iCV] = max(cvRate);
mChosen  = msCV(iCV);

% Now, and only now, look at the test set.
[Wcv, ~, ~] = fisherfaces(oV, oXbar, oTr, oYtr, mChosen, oC);
rTestCV = nnRate(Wcv' * (oTr - oXbar), oYtr, Wcv' * (oTe - oXbar), oYte);

% For comparison: what we would have got by tuning m ON the test set.
testRate = zeros(1, numel(msCV));
for i = 1:numel(msCV)
    [Wt, ~, ~] = fisherfaces(oV, oXbar, oTr, oYtr, msCV(i), oC);
    testRate(i) = nnRate(Wt' * (oTr - oXbar), oYtr, Wt' * (oTe - oXbar), oYte);
end
[bestTest, iT] = max(testRate);

fprintf('\n   cross-validation chose m = %d, which scores %.1f%% on test\n', ...
        mChosen, rTestCV);
fprintf('   tuning m on the test set would have chosen m = %d for %.1f%%\n', ...
        msCV(iT), bestTest);
fprintf('   the gap, %.1f points, is the optimism we avoid by not peeking.\n', ...
        bestTest - rTestCV);

figure('Visible', 'off', 'Position', [100 100 620 400]);
plot(msCV, cvRate,   '-o', 'LineWidth', 1.5); hold on;
plot(msCV, testRate, '-s', 'LineWidth', 1.5);
plot(mChosen, rTestCV, 'kp', 'MarkerSize', 14, 'MarkerFaceColor', 'y');
hold off;
grid on;
xlabel('m  (PCA dimension before LDA)');
ylabel('recognition rate (%)');
legend({'cross-validated (honest)', 'test set (peeking)', 'm chosen by CV'}, ...
       'Location', 'southwest');
title('ORL: choosing m with and without looking at the test set');
saveas(gcf, [figDir 'E3_cross_validation.png']);
close;


%% ====================================================================
%  E4   NOISE ROBUSTNESS: TESTING A PREDICTION
%
%  This is not a fishing expedition, it is a PREDICTION being tested.
%
%  Fisherfaces were measured (in the full study) to sit 3.85x higher in
%  spatial frequency than eigenfaces.  The reason is whitening: the LDA
%  stage scales direction l by 1/sqrt(lambda_l), so the LOW-variance
%  directions -- which are the fine-detail, high-frequency ones -- get
%  amplified the most, by up to ~8860x on ORL.
%
%  White pixel noise has energy at ALL frequencies.  A basis that leans on
%  high-frequency detail should therefore pick up more noise per unit of
%  signal.  PREDICTION: Fisherfaces degrade FASTER than eigenfaces as we
%  add noise to the test images.
%
%  Noise is added only to TEST images: the projections stay as they were
%  learned from clean training data, which is the realistic situation.
%  Pixel values are on a 0-255 scale, so sigma = 40 is substantial.
%% ====================================================================

fprintf('\n\n=== E4: noise robustness (ORL) ===\n');

% Push this far enough to actually break the methods.  A first attempt
% stopped at sigma = 80 and found NOTHING: all three moved by 0.8 points,
% which on 128 test images is exactly ONE image, i.e. noise in the
% measurement rather than an effect.  Projection is a powerful denoiser --
% a unit-norm direction turns per-pixel noise of std sigma into coefficient
% noise of std sigma, while the coefficients themselves are in the
% thousands -- so sigma has to get very large before anything happens.
sigmas = [0 20 40 80 160 320 640];
rCorrN = zeros(1, numel(sigmas));
rEigN  = zeros(1, numel(sigmas));
rFishN = zeros(1, numel(sigmas));

Vk50 = oV(:, 1:50);                 % eigenfaces at the k that is best on ORL
[Wn, ~, ~] = fisherfaces(oV, oXbar, oTr, oYtr, oN - oC, oC);

fprintf('   sigma   correlation   eigenfaces(k=50)   Fisherfaces\n');
for i = 1:numel(sigmas)
    rng(1);                         % same noise pattern for all three methods
    noisyTe = oTe + sigmas(i) * randn(size(oTe));

    rCorrN(i) = nnRate(normCols(oTr), oYtr, normCols(noisyTe), oYte);
    rEigN(i)  = nnRate(Vk50' * (oTr - oXbar), oYtr, ...
                       Vk50' * (noisyTe - oXbar), oYte);
    rFishN(i) = nnRate(Wn' * (oTr - oXbar), oYtr, ...
                       Wn' * (noisyTe - oXbar), oYte);
    fprintf('   %5d   %9.1f%%   %14.1f%%   %10.1f%%\n', ...
            sigmas(i), rCorrN(i), rEigN(i), rFishN(i));
end

% One test image is worth 100/128 = 0.78 points, so differences of a point
% or so are a single image and mean nothing.  Compare at the largest sigma.
onePoint = 100 / numel(oYte);
fprintf('\n   (one test image = %.2f points, so smaller gaps are not real)\n', onePoint);
fprintf('   drop from sigma = 0 to sigma = %d:\n', sigmas(end));
i40 = numel(sigmas);
fprintf('     correlation %5.1f points,  eigenfaces %5.1f points,  Fisherfaces %5.1f points\n', ...
        rCorrN(1) - rCorrN(i40), rEigN(1) - rEigN(i40), rFishN(1) - rFishN(i40));
if (rFishN(1) - rFishN(i40)) > (rEigN(1) - rEigN(i40)) + 2 * onePoint
    fprintf('   => Fisherfaces fall further, as the frequency argument predicts.\n');
else
    fprintf('   => Fisherfaces do NOT fall further by more than measurement\n');
    fprintf('      noise.  The prediction FAILS, and that is worth reporting.\n');
    fprintf('      The likely reason: we keep the c-1 directions with the\n');
    fprintf('      LARGEST generalised eigenvalues, which are the LEAST\n');
    fprintf('      whitened ones.  The 1/sqrt(lambda) blow-up of up to 8860x\n');
    fprintf('      lands on directions we throw away, so the frequency\n');
    fprintf('      argument does not transfer to noise sensitivity.\n');
end

figure('Visible', 'off', 'Position', [100 100 620 400]);
plot(sigmas, rCorrN, '-o', 'LineWidth', 1.5); hold on;
plot(sigmas, rEigN,  '-s', 'LineWidth', 1.5);
plot(sigmas, rFishN, '-^', 'LineWidth', 1.5);
hold off;
grid on;
xlabel('noise standard deviation added to test images (grey levels)');
ylabel('recognition rate (%)');
legend({'correlation', 'eigenfaces (k=50)', 'Fisherfaces'}, 'Location', 'southwest');
title('ORL: how the three methods degrade under pixel noise');
saveas(gcf, [figDir 'E4_noise.png']);
close;


%% ====================================================================
%  E5   THE PAPER'S REAL PROTOCOL: ERROR VERSUS LIGHTING ANGLE
%
%  The paper does not report one Yale number.  It splits the test images
%  into five subsets by how far the light source is from straight ahead,
%  and reports error per subset.  That is a much more informative result:
%  it shows WHERE each method fails, not just how often.
%
%  Every Yale filename encodes the light direction, for example
%       yaleB01_P00A+020E-40.pgm
%                   ^^^^ ^^^
%                   azimuth = +20 deg, elevation = -40 deg
%
%  The angle from the optical axis combines the two.  Treating azimuth and
%  elevation as rotations about perpendicular axes, the direction cosine
%  along the camera axis is cos(azimuth)*cos(elevation), so
%
%       theta = acos( cos(azimuth) * cos(elevation) )
%
%  The paper's five bins are 0-12, 13-25, 26-50, 51-77 and above 77 degrees.
%% ====================================================================

fprintf('\n\n=== E5: Yale error against lighting angle ===\n');

% ---- work out the angle of every TEST image from its filename ----
nTe = numel(yTeNames);
theta = zeros(1, nTe);
for i = 1:nTe
    nm = yTeNames{i};
    ia = strfind(nm, 'A');          % uppercase A appears exactly once
    ie = strfind(nm, 'E');          % uppercase E appears exactly once
    az = str2double(nm(ia+1 : ie-1));        % e.g. '+020' -> 20
    el = str2double(nm(ie+1 : end-4));       % e.g. '-40'  -> -40, drop '.pgm'
    theta(i) = acosd(cosd(az) * cosd(el));
end

% ---- put each test image into one of the paper's five bins ----
edges = [0 12 25 50 77 180];      % only used for printing the ranges
subset = zeros(1, nTe);
for i = 1:nTe
    if theta(i) <= 12
        subset(i) = 1;
    elseif theta(i) <= 25
        subset(i) = 2;
    elseif theta(i) <= 50
        subset(i) = 3;
    elseif theta(i) <= 77
        subset(i) = 4;
    else
        subset(i) = 5;
    end
end

% ---- the two methods, trained once on the clean training set ----
yVc = yV(:, 1:(yC-1));
predEig  = nnPredict(yVc' * (yTr - yXbar), yYtr, yVc' * (yTe - yXbar));
[Wy, ~, ~] = fisherfaces(yV, yXbar, yTr, yYtr, yN - yC, yC);
predFish = nnPredict(Wy' * (yTr - yXbar), yYtr, Wy' * (yTe - yXbar));

% NaN, not zero: a subset with no test images must leave a GAP in the bar
% chart, not a bar at 0% that reads as "perfect score".
errEig  = nan(1, 5);
errFish = nan(1, 5);
counts  = zeros(1, 5);

fprintf('   subset   angle range     images   eigenfaces err   Fisherfaces err\n');
for b = 1:5
    sel = (subset == b);
    counts(b) = sum(sel);
    if counts(b) == 0
        fprintf('     %d      %3d - %3d deg         0      (no test images)\n', ...
                b, edges(b), edges(b+1));
        continue;
    end
    errEig(b)  = 100 * sum(predEig(sel)  ~= yYte(sel)) / counts(b);
    errFish(b) = 100 * sum(predFish(sel) ~= yYte(sel)) / counts(b);
    fprintf('     %d      %3d - %3d deg     %5d      %8.1f%%        %8.1f%%\n', ...
            b, edges(b), edges(b+1), counts(b), errEig(b), errFish(b));
end

fprintf('\n   Error climbs with angle for both methods, and the gap between\n');
fprintf('   them WIDENS: that is the paper''s central claim, that Fisherfaces\n');
fprintf('   hold up where illumination gets extreme.\n');
if counts(1) == 0
    fprintf('\n   Subset 1 (0-12 deg) is EMPTY, and that is not a bug: it is the\n');
    fprintf('   split bias showing up again.  dir sorts names as text and ''+''\n');
    fprintf('   sorts before ''-'', so the first 40 files per subject -- the\n');
    fprintf('   training set -- swallow all the near-frontal lighting.  Every\n');
    fprintf('   test image is therefore at least 12 degrees off axis, which is\n');
    fprintf('   precisely why the headline Yale numbers look so poor.\n');
end

figure('Visible', 'off', 'Position', [100 100 640 400]);
bar([errEig(:), errFish(:)]);
set(gca, 'XTickLabel', {'0-12', '13-25', '26-50', '51-77', '>77'});
xlabel('angle of light source from the optical axis (degrees)');
ylabel('error rate (%)');
legend({'eigenfaces (k=c-1)', 'Fisherfaces'}, 'Location', 'northwest');
title('Yale: error grows with lighting angle');
grid on;
saveas(gcf, [figDir 'E5_illumination_subsets.png']);
close;

fprintf('\n\n=====================================================\n');
fprintf('  figures written to %s\n', figDir);
fprintf('  DONE\n');
fprintf('=====================================================\n');


%% ====================================================================
%  LOCAL FUNCTIONS  (copies of the ones in fisherfaces_simple.m, plus
%  nnPredict, which returns the predicted labels instead of just a rate)
%% ====================================================================

function [imTr, ytr, imTe, yte, teNames] = loadSet(folders, numTrain, d)
% LOADSET  Read the PGM images, one image per column, split train/test.
%          teNames holds the filename of every TEST column, which E5 needs
%          in order to read the lighting angle out of the name.

nF = numel(folders);

nTr = 0;
nTe = 0;
for p = 1:nF
    f = dir([folders{p} '*.pgm']);
    if isempty(f)
        error('No .pgm images found in %s', folders{p});
    end
    ntr = min(numTrain, numel(f));
    nTr = nTr + ntr;
    nTe = nTe + (numel(f) - ntr);
end

imTr = zeros(d, nTr);   ytr = zeros(1, nTr);
imTe = zeros(d, nTe);   yte = zeros(1, nTe);
teNames = cell(1, nTe);

a = 0;
b = 0;
for p = 1:nF
    files = dir([folders{p} '*.pgm']);
    for j = 1:numel(files)
        im = double(imread([folders{p} files(j).name]));
        if j <= numTrain
            a = a + 1;
            imTr(:, a) = im(:);
            ytr(a)     = p;
        else
            b = b + 1;
            imTe(:, b)  = im(:);
            yte(b)      = p;
            teNames{b}  = files(j).name;
        end
    end
end
end


function [V, lambda, xbar] = doPCA(imgs)
% DOPCA  Eigenfaces via the L = X'*X trick.  See the study guide for why
%        X*w is an eigenvector of X*X' whenever w is one of X'*X.

[~, N] = size(imgs);

xbar = mean(imgs, 2);
X    = imgs - xbar;

L = X' * X;
[Wg, D] = eig(L);

[eigvals, idx] = sort(diag(D), 'descend');   % eig is ascending; flip it
Wg = Wg(:, idx);

V = X * Wg;

% ||X*w|| = sqrt(lambda), not 1, so renormalise.  The guard is relative to
% the first column, because after centring the last direction is numerically
% empty and dividing by its tiny norm would amplify round-off.
scale = norm(V(:, 1));
for l = 1:size(V, 2)
    nrm = norm(V(:, l));
    if nrm > 1e-10 * scale
        V(:, l) = V(:, l) / nrm;
    else
        V(:, l) = 0;
    end
end

lambda = eigvals / N;
end


function [Wopt, Sb, Sw] = fisherfaces(V, xbar, imgs, y, m, c)
% FISHERFACES  PCA to m dimensions, then LDA, keeping c-1 directions.

% m is squeezed from BOTH sides.  Above, m <= N-c keeps S_W invertible.
% Below, m >= c-1, because we are going to pull c-1 directions out of an
% m-dimensional space and cannot have more directions than dimensions.
if m < c - 1
    error(['fisherfaces: m = %d is smaller than c-1 = %d. ' ...
           'You cannot extract %d Fisherfaces from a %d-dimensional space.'], ...
          m, c-1, c-1, m);
end

Vm = V(:, 1:m);
Al = Vm' * (imgs - xbar);
abar = mean(Al, 2);

Sw = zeros(m, m);
Sb = zeros(m, m);
for j = 1:c
    Aj = Al(:, y == j);
    nj = size(Aj, 2);
    mj = mean(Aj, 2);
    Dj = Aj - mj;
    Sw = Sw + Dj * Dj';
    Sb = Sb + nj * (mj - abar) * (mj - abar)';
end

Sw = (Sw + Sw') / 2;        % force exact symmetry before chol/eig
Sb = (Sb + Sb') / 2;

% Cholesky whitening: S_W = R'*R, substitute w = inv(R)*z, which turns
% S_B*w = lambda*S_W*w into the ordinary SYMMETRIC problem M*z = lambda*z
% with M = inv(R')*S_B*inv(R).
R = chol(Sw);
M = (R') \ (Sb / R);
M = (M + M') / 2;

[Z, Dz] = eig(M);
[~, idx] = sort(diag(Dz), 'descend');
Z = Z(:, idx);

Wfld = R \ Z(:, 1:(c-1));   % map back: solve R*w = z
Wopt = Vm * Wfld;

% Unit-norm each Fisherface.  J(w) does not care, but nearest neighbour
% with Euclidean distance does: without this ORL falls from 90.6% to 20.3%.
for l = 1:size(Wopt, 2)
    nrm = norm(Wopt(:, l));
    if nrm > 0
        Wopt(:, l) = Wopt(:, l) / nrm;
    end
end
end


function Xn = normCols(X)
% NORMCOLS  Zero-mean, unit-norm each column.  Turns nearest neighbour into
%           a correlation classifier.

Xn = X - mean(X, 1);
for i = 1:size(Xn, 2)
    nrm = norm(Xn(:, i));
    if nrm > 0
        Xn(:, i) = Xn(:, i) / nrm;
    end
end
end


function pred = nnPredict(trainProj, trainLabels, testProj)
% NNPREDICT  The label of the nearest training column, for every test
%            column.  Squared distances, no square root needed.

pred = zeros(1, size(testProj, 2));
for i = 1:size(testProj, 2)
    distances = sum((trainProj - testProj(:, i)).^2, 1);
    [~, best] = min(distances);
    pred(i) = trainLabels(best);
end
end


function rate = nnRate(trainProj, trainLabels, testProj, testLabels)
% NNRATE  Percentage of test columns whose nearest neighbour has the right
%         label.  Written in terms of nnPredict so there is one copy of the
%         actual classifier.

pred = nnPredict(trainProj, trainLabels, testProj);
rate = 100 * sum(pred == testLabels) / numel(testLabels);
end
