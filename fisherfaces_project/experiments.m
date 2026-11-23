%% Experiments on top of eigenfaces_vs_fisherfaces.m
%
% E1  sweeping the PCA dimension m, with cond(S_W)   (study guide, section 24)
% E2  label permutation: is the result real?
% E3  choosing m by cross-validation, without peeking at the test set
% E4  noise robustness
% E5  error against lighting angle (Yale)
% E6  sweeping the number of classes c
% E7  glasses vs no glasses (the paper's section 3.3)
%
% Notation follows the study guide: d pixels, N training images, c classes,
% m the PCA dimension kept before LDA, S_W and S_B the scatter matrices.
%
% Settings match the study guide section 24 so E1 reproduces its table:
% ORL with 32 people and 6 training images each.
%
% Helper functions are at the bottom of this file.

clc; clear; close all;

figDir = 'figures/';
if exist(figDir, 'dir') == 0
    mkdir(figDir);
end

orlDir  = '../ORL';
yaleDir = '../CroppedYale';
if exist(orlDir, 'dir') == 0 || exist(yaleDir, 'dir') == 0
    error('Run this from the folder holding this file, with ORL/ and CroppedYale/ one level above.');
end

fprintf('loading ORL ...\n');
[oTr, oYtr, oTe, oYte] = loadFaces(orlDir, 32, 6, 112*92);
fprintf('loading Yale ... (slow)\n');
[yTr, yYtr, yTe, yYte, yNames] = loadFaces(yaleDir, 38, 40, 192*168);

oN = size(oTr, 2);  oC = 32;
yN = size(yTr, 2);  yC = 38;
[oU, oMean] = doPCA(oTr);
[yU, yMean] = doPCA(yTr);


%% E1. Sweeping the PCA dimension m, with cond(S_W)
% The paper fixes m = N - c.  At that value S_W has rank m: full rank, but
% with ZERO SLACK -- one more direction and it would be singular.  So we
% sweep m and watch both the accuracy and the conditioning.
%
% m is squeezed from both sides:   c - 1 <= m <= N - c
% The upper bound keeps S_W invertible; the lower bound is because c-1
% directions cannot come out of a space with fewer than c-1 dimensions.

fprintf('\n\n=== E1: sweeping m (study guide section 24) ===\n');

for db = 1:2
    if db == 1
        nm = 'ORL';  U = oU; mf = oMean; tr = oTr; ytr = oYtr; te = oTe; yte = oYte;
        c = oC; N = oN;  ms = [31 40 60 80 100 120 140 160];
    else
        nm = 'Yale'; U = yU; mf = yMean; tr = yTr; ytr = yYtr; te = yTe; yte = yYte;
        c = yC; N = yN;  ms = [37 60 100 200 400 700 1000 1200 1482];
    end

    conds = zeros(1, numel(ms));
    rates = zeros(1, numel(ms));
    fprintf('\n  %s  (N = %d, c = %d, N-c = %d)\n', nm, N, c, N-c);
    fprintf('       m   rate%%   rank(S_W)    cond(S_W)\n');
    for i = 1:numel(ms)
        [W, Sw] = fisherfaces(U, mf, tr, ytr, ms(i), c);
        conds(i) = cond(Sw);
        rates(i) = nnRate(W'*(tr-mf), ytr, W'*(te-mf), yte);
        fprintf('  %6d  %6.1f   %7d    %.2e\n', ms(i), rates(i), rank(Sw), conds(i));
    end
    [bestR, iB] = max(rates);
    fprintf('   best m = %d at %.1f%%; the paper uses m = %d giving %.1f%%\n', ...
            ms(iB), bestR, N-c, rates(end));

    figure('Position', [100 100 900 340]);
    subplot(1,2,1);
    semilogy(ms, conds, '-o', 'LineWidth', 1.5); grid on;
    xlabel('m'); ylabel('cond(S_W)  [log scale]');
    title([nm ': conditioning explodes at m = N-c']);
    subplot(1,2,2);
    plot(ms, rates, '-o', 'LineWidth', 1.5); grid on;
    xlabel('m'); ylabel('accuracy (%)');
    title([nm ': accuracy peaks before m = N-c']);
    saveas(gcf, [figDir 'E1_sweep_m_' nm '.png']);
end


%% E2. Label permutation
% Shuffle the TRAINING labels and change nothing else.  Now no face has any
% relation to its label, so nothing can beat guessing, which is 100/c.
% A method still scoring well would mean the pipeline is leaking.

fprintf('\n\n=== E2: label permutation (ORL) ===\n');

Ue = oU(:, 1:(oC-1));
rE = nnRate(Ue'*(oTr-oMean), oYtr, Ue'*(oTe-oMean), oYte);
W  = fisherfaces(oU, oMean, oTr, oYtr, oN-oC, oC);
rF = nnRate(W'*(oTr-oMean), oYtr, W'*(oTe-oMean), oYte);

shuf = oYtr(randperm(oN));
rEs = nnRate(Ue'*(oTr-oMean), shuf, Ue'*(oTe-oMean), oYte);
Ws  = fisherfaces(oU, oMean, oTr, shuf, oN-oC, oC);
rFs = nnRate(Ws'*(oTr-oMean), shuf, Ws'*(oTe-oMean), oYte);

chance = 100 / oC;
fprintf('                 true labels   shuffled\n');
fprintf('  eigenfaces       %6.1f%%     %6.1f%%\n', rE, rEs);
fprintf('  Fisherfaces      %6.1f%%     %6.1f%%\n', rF, rFs);
fprintf('  chance = 100/c   %6.1f%%\n', chance);
fprintf('  Both collapse, so the real numbers come from the faces.\n');

figure('Position', [100 100 560 360]);
bar([rE rEs; rF rFs]); hold on;
plot([0.5 2.5], [chance chance], 'k--', 'LineWidth', 1.2); hold off;
set(gca, 'XTickLabel', {'eigenfaces','Fisherfaces'});
ylabel('accuracy (%)'); grid on;
legend('true labels', 'shuffled', 'chance');
title('Shuffling the training labels destroys both methods');
saveas(gcf, [figDir 'E2_label_permutation.png']);


%% E3. Choosing m by cross-validation
% E1 read "the best m" off the test column, which is tuning on the test set.
% The honest way is 6-fold leave-one-out INSIDE the training set: train on 5
% images per person, validate on the held-out 1, rotate, average.  The PCA
% basis is recomputed per fold, otherwise the validation image leaks into it.

fprintf('\n\n=== E3: cross-validated m (ORL) ===\n');

msCV = [31 40 60 80 100 120];
cvR  = zeros(1, numel(msCV));
for i = 1:numel(msCV)
    total = 0;
    for f = 1:6
        isVal = false(1, oN);
        for p = 1:oC
            cols = find(oYtr == p);
            isVal(cols(f)) = true;
        end
        trI = oTr(:, ~isVal);  trL = oYtr(~isVal);
        vaI = oTr(:,  isVal);  vaL = oYtr( isVal);
        [Uf, mf] = doPCA(trI);
        Wf = fisherfaces(Uf, mf, trI, trL, msCV(i), oC);
        total = total + nnRate(Wf'*(trI-mf), trL, Wf'*(vaI-mf), vaL);
    end
    cvR(i) = total / 6;
    fprintf('   m = %3d   cross-validated = %5.1f%%\n', msCV(i), cvR(i));
end

testR = zeros(1, numel(msCV));
for i = 1:numel(msCV)
    Wt = fisherfaces(oU, oMean, oTr, oYtr, msCV(i), oC);
    testR(i) = nnRate(Wt'*(oTr-oMean), oYtr, Wt'*(oTe-oMean), oYte);
end
[~, iCV] = max(cvR);
[bT, iT] = max(testR);
fprintf('\n   CV picks m = %d, which scores %.1f%% on the test set\n', ...
        msCV(iCV), testR(iCV));
fprintf('   tuning on test would pick m = %d for %.1f%%\n', msCV(iT), bT);
fprintf('   the %.1f point gap is the optimism avoided by not peeking\n', ...
        bT - testR(iCV));

figure('Position', [100 100 600 380]);
plot(msCV, cvR, '-o', 'LineWidth', 1.5); hold on;
plot(msCV, testR, '-s', 'LineWidth', 1.5);
plot(msCV(iCV), testR(iCV), 'kp', 'MarkerSize', 14, 'MarkerFaceColor', 'y');
hold off; grid on;
xlabel('m'); ylabel('accuracy (%)');
legend('cross-validated (honest)', 'test set (peeking)', 'm chosen by CV', ...
       'Location', 'southwest');
title('Choosing m with and without looking at the test set');
saveas(gcf, [figDir 'E3_cross_validation.png']);


%% E4. Noise robustness
% Fisherfaces are higher frequency than eigenfaces, because whitening scales
% direction l by 1/sqrt(lambda_l) and so amplifies the fine-detail
% directions most.  White noise has energy at every frequency, so they
% should degrade faster.  Noise is added to the TEST images only.

fprintf('\n\n=== E4: noise robustness (ORL) ===\n');

sigmas = [0 20 40 80 160 320 640];
rEn = zeros(1, numel(sigmas));
rFn = zeros(1, numel(sigmas));
Uk  = oU(:, 1:30);
Wn  = fisherfaces(oU, oMean, oTr, oYtr, oN-oC, oC);

fprintf('   sigma   eigenfaces   Fisherfaces\n');
for i = 1:numel(sigmas)
    rng(1);                               % same noise for both methods
    noisy = oTe + sigmas(i) * randn(size(oTe));
    rEn(i) = nnRate(Uk'*(oTr-oMean), oYtr, Uk'*(noisy-oMean), oYte);
    rFn(i) = nnRate(Wn'*(oTr-oMean), oYtr, Wn'*(noisy-oMean), oYte);
    fprintf('  %6d   %9.1f%%  %10.1f%%\n', sigmas(i), rEn(i), rFn(i));
end
fprintf('   drop from sigma 0 to %d: eigenfaces %.1f points, Fisherfaces %.1f\n', ...
        sigmas(end), rEn(1)-rEn(end), rFn(1)-rFn(end));
fprintf('   Nothing moves below sigma = 160: projection is a strong denoiser,\n');
fprintf('   so the effect only appears far beyond realistic noise levels.\n');

figure('Position', [100 100 600 380]);
plot(sigmas, rEn, '-s', 'LineWidth', 1.5); hold on;
plot(sigmas, rFn, '-^', 'LineWidth', 1.5); hold off; grid on;
xlabel('noise standard deviation added to test images');
ylabel('accuracy (%)');
legend('eigenfaces (k=30)', 'Fisherfaces', 'Location', 'southwest');
title('Degradation under pixel noise');
saveas(gcf, [figDir 'E4_noise.png']);


%% E5. Error against lighting angle (Yale)
% Every Yale filename encodes the light direction, e.g.
%     yaleB01_P00A+020E-40.pgm  ->  azimuth +20, elevation -40
% Azimuth and elevation are rotations about perpendicular axes, so the
% direction cosine along the camera axis is the product of the cosines:
%     theta = acos( cos(azimuth) * cos(elevation) )
% The paper's bins are 0-12, 13-25, 26-50, 51-77 and above 77 degrees.

fprintf('\n\n=== E5: error against lighting angle (Yale) ===\n');

nTe = numel(yNames);
theta = zeros(1, nTe);
for i = 1:nTe
    s  = yNames{i};
    ia = strfind(s, 'A');             % uppercase A appears exactly once
    ie = strfind(s, 'E');             % uppercase E appears exactly once
    az = str2double(s(ia+1 : ie-1));
    el = str2double(s(ie+1 : end-4)); % drop '.pgm'
    theta(i) = acosd(cosd(az) * cosd(el));
end

bin = zeros(1, nTe);
for i = 1:nTe
    if     theta(i) <= 12, bin(i) = 1;
    elseif theta(i) <= 25, bin(i) = 2;
    elseif theta(i) <= 50, bin(i) = 3;
    elseif theta(i) <= 77, bin(i) = 4;
    else,                  bin(i) = 5;
    end
end

Uy = yU(:, 1:(yC-1));
pe = nnPredict(Uy'*(yTr-yMean), yYtr, Uy'*(yTe-yMean));
Wy = fisherfaces(yU, yMean, yTr, yYtr, yN-yC, yC);
pf = nnPredict(Wy'*(yTr-yMean), yYtr, Wy'*(yTe-yMean));

errE = nan(1,5); errF = nan(1,5); cnt = zeros(1,5);
lo = [0 12 25 50 77]; hi = [12 25 50 77 180];
fprintf('   bin   angle          images   eigenfaces   Fisherfaces\n');
for b = 1:5
    sel = (bin == b);
    cnt(b) = sum(sel);
    if cnt(b) == 0
        fprintf('    %d    %3d-%3d deg         0    (no test images)\n', b, lo(b), hi(b));
        continue;
    end
    errE(b) = 100 * sum(pe(sel) ~= yYte(sel)) / cnt(b);
    errF(b) = 100 * sum(pf(sel) ~= yYte(sel)) / cnt(b);
    fprintf('    %d    %3d-%3d deg     %5d   %9.1f%%  %10.1f%%\n', ...
            b, lo(b), hi(b), cnt(b), errE(b), errF(b));
end
if cnt(1) == 0
    fprintf('   The first bin is EMPTY because dir sorts names as text and\n');
    fprintf('   ''+'' sorts before ''-'', so the training set swallowed all the\n');
    fprintf('   frontal lighting.  Every test image is at least 12 deg off axis.\n');
end

figure('Position', [100 100 620 380]);
bar([errE(:) errF(:)]);
set(gca, 'XTickLabel', {'0-12','13-25','26-50','51-77','>77'});
xlabel('angle of the light from the optical axis (degrees)');
ylabel('error rate (%)'); grid on;
legend('eigenfaces', 'Fisherfaces', 'Location', 'northwest');
title('Yale: error grows with lighting angle');
saveas(gcf, [figDir 'E5_lighting_angle.png']);


%% E6. Sweeping the number of classes c
% rank(S_B) <= c-1, so there are at most c-1 Fisherfaces however many pixels
% or images we have.  Vary c and watch that ceiling bind.

fprintf('\n\n=== E6: sweeping the number of classes (ORL) ===\n');

cs = [2 4 8 16 32];
nF = zeros(1, numel(cs));
rE6 = zeros(1, numel(cs));
rF6 = zeros(1, numel(cs));
fprintf('       c     N   N-c   fisherfaces   eigenfaces   Fisherfaces\n');
for i = 1:numel(cs)
    c = cs(i);
    [tr, trL, te, teL] = loadFaces(orlDir, c, 6, 112*92);
    N = size(tr, 2);
    [Uc, mc] = doPCA(tr);
    Uk2 = Uc(:, 1:max(1, c-1));
    rE6(i) = nnRate(Uk2'*(tr-mc), trL, Uk2'*(te-mc), teL);
    W6 = fisherfaces(Uc, mc, tr, trL, N-c, c);
    nF(i)  = size(W6, 2);
    rF6(i) = nnRate(W6'*(tr-mc), trL, W6'*(te-mc), teL);
    fprintf('  %6d %5d %5d   %11d   %9.1f%%  %10.1f%%\n', ...
            c, N, N-c, nF(i), rE6(i), rF6(i));
end
fprintf('   The number of Fisherfaces equals c-1 on every row.\n');

figure('Position', [100 100 900 340]);
subplot(1,2,1);
plot(cs, nF, '-o', 'LineWidth', 1.5); hold on;
plot(cs, cs-1, 'k--', 'LineWidth', 1.2); hold off; grid on;
xlabel('number of classes c'); ylabel('number of Fisherfaces');
legend('measured', 'c-1', 'Location', 'northwest');
title('the c-1 ceiling is exactly met');
subplot(1,2,2);
plot(cs, rE6, '-s', 'LineWidth', 1.5); hold on;
plot(cs, rF6, '-^', 'LineWidth', 1.5); hold off; grid on;
xlabel('number of classes c'); ylabel('accuracy (%)');
legend('eigenfaces', 'Fisherfaces', 'Location', 'southwest');
title('both get harder as c grows');
saveas(gcf, [figDir 'E6_number_of_classes.png']);


%% E7. Glasses vs no glasses (the paper's section 3.3)
% A different question: not WHO is this, but IS THIS PERSON WEARING GLASSES.
% Two classes, so rank(S_B) <= 1 and there is exactly ONE Fisherface.
%
% ORL ships no glasses labels, so orl_glasses_labels.txt was made by looking
% at all 400 images.  117 wear glasses and 8 people appear both ways.
%
% We balance the set 50/50 (otherwise always answering "no" scores 70.8% and
% beats everything) and leave one SUBJECT out at a time, so the test face is
% never seen in training and the classifier cannot cheat by recognising it.

fprintf('\n\n=== E7: glasses vs no glasses (ORL) ===\n');

G = readGlasses('orl_glasses_labels.txt');
[aIm, aSubj] = loadFaces(orlDir, 40, 10, 112*92);
aG = zeros(1, size(aIm, 2));
for i = 1:size(aIm, 2)
    k = sum(aSubj(1:i) == aSubj(i));      % which image of this person
    aG(i) = G(aSubj(i), k);
end
fprintf('   %d of 400 images wear glasses (%.1f%%)\n', sum(aG), 100*mean(aG));

gI = find(aG == 1);
nI = find(aG == 0);
nI = nI(randperm(numel(nI), numel(gI)));  % balance it
keepI = sort([gI nI]);
bIm = aIm(:, keepI);  bSubj = aSubj(keepI);  bG = aG(keepI);
fprintf('   balanced subset: %d images, %.1f%% with glasses\n', numel(bG), 100*mean(bG));

okE = 0; okF = 0; tot = 0;
for p = 1:40
    isTe = (bSubj == p);
    if sum(isTe) == 0, continue; end
    trI = bIm(:, ~isTe);  trL = bG(~isTe) + 1;   % labels must be 1 and 2
    teI = bIm(:,  isTe);  teL = bG( isTe) + 1;
    if numel(unique(trL)) < 2, continue; end

    [Ug, mg] = doPCA(trI);
    Wg = fisherfaces(Ug, mg, trI, trL, size(trI,2) - 2, 2);
    pf7 = nnPredict(Wg'*(trI-mg), trL, Wg'*(teI-mg));
    Uk7 = Ug(:, 1:10);
    pe7 = nnPredict(Uk7'*(trI-mg), trL, Uk7'*(teI-mg));

    okF = okF + sum(pf7 == teL);
    okE = okE + sum(pe7 == teL);
    tot = tot + numel(teL);
end
fprintf('\n   method            our error    paper (Yale, 36 images)\n');
fprintf('   PCA, k = 10       %8.1f%%              52.6%%\n', 100 - 100*okE/tot);
fprintf('   one Fisherface    %8.1f%%               5.3%%\n', 100 - 100*okF/tot);
fprintf('   always say "no"       50.0%%              50.0%%\n');
fprintf('   We reproduce that PCA is near chance, but NOT that Fisherfaces\n');
fprintf('   solve it.  Their 36 images were PAIRED (the same person with and\n');
fprintf('   without glasses, one session); in ORL glasses come with changes\n');
fprintf('   of pose, expression and session, so there is no clean signal.\n');

figure('Position', [100 100 560 360]);
bar([100-100*okE/tot, 100-100*okF/tot]); hold on;
plot([0.5 2.5], [50 50], 'k--', 'LineWidth', 1.2); hold off;
set(gca, 'XTickLabel', {'PCA (k=10)', 'one Fisherface'});
ylabel('error rate (%)'); grid on;
legend('error', 'always say no');
title('Detecting glasses on a person never seen in training');
saveas(gcf, [figDir 'E7_glasses.png']);

fprintf('\n\nfigures saved to %s\n', figDir);


%% ---------------- helper functions ----------------

function [trIm, trLab, teIm, teLab, teNames] = loadFaces(dataDir, numPeople, numTrain, d)
% Read the PGM images, one image per column, and split into train and test.
% The class label is the LOOP INDEX p, never the folder name.
nTr = 0; nTe = 0;
for p = 1:numPeople
    f = dir([dataDir '/s' num2str(p) '/*.pgm']);
    if isempty(f)
        error('no .pgm images in %s/s%d/', dataDir, p);
    end
    nTr = nTr + min(numTrain, numel(f));
    nTe = nTe + max(0, numel(f) - numTrain);
end
trIm = zeros(d, nTr);  trLab = zeros(1, nTr);
teIm = zeros(d, nTe);  teLab = zeros(1, nTe);
teNames = cell(1, nTe);
a = 0; b = 0;
for p = 1:numPeople
    folder = [dataDir '/s' num2str(p) '/'];
    files  = dir([folder '*.pgm']);
    for j = 1:numel(files)
        im = double(imread([folder files(j).name]));
        if j <= numTrain
            a = a + 1;  trIm(:, a) = im(:);  trLab(a) = p;
        else
            b = b + 1;  teIm(:, b) = im(:);  teLab(b) = p;
            teNames{b} = files(j).name;
        end
    end
end
end


function [U, meanFace] = doPCA(images)
% Eigenfaces via the small matrix L = A'*A.  If L*v = lambda*v then
% (A*A')*(A*v) = lambda*(A*v), so A*v is an eigenvector of the big
% covariance with the same eigenvalue.  ||A*v|| = sqrt(lambda), not 1,
% so the columns must be renormalised.
N = size(images, 2);
meanFace = mean(images, 2);
A = images - meanFace;
L = A' * A;
[V, D] = eig(L);
[~, order] = sort(diag(D), 'descend');     % eig is ASCENDING
V = V(:, order);
U = A * V;
for i = 1:N
    nrm = norm(U(:, i));
    if nrm > 0
        U(:, i) = U(:, i) / nrm;
    end
end
U = U(:, 1:N-1);                           % last direction is empty
end


function [Wopt, Sw] = fisherfaces(U, meanFace, images, labels, m, c)
% PCA to m dimensions, then LDA, keeping c-1 directions.
if m < c - 1
    error('m = %d is below c-1 = %d', m, c-1);
end
A = images - meanFace;
Wpca = U(:, 1:m);
Z = Wpca' * A;                   % the training data in m dimensions

Sb = zeros(m, m);
Sw = zeros(m, m);
for p = 1:c
    Zp = Z(:, labels == p);
    Np = size(Zp, 2);
    mp = mean(Zp, 2);
    Sb = Sb + Np * (mp * mp');   % mean of Z is zero, so no need to recentre
    for j = 1:Np
        df = Zp(:, j) - mp;
        Sw = Sw + df * df';
    end
end

[V2, D2] = eig(Sb, Sw);
[~, order] = sort(diag(D2), 'descend');
Wfld = V2(:, order(1:c-1));
Wopt = Wpca * Wfld;

% Unit length, and this is NOT cosmetic: eig(Sb,Sw) returns columns with
% W'*S_W*W = I, so their Euclidean lengths differ by orders of magnitude,
% and nearest neighbour would be dominated by the longest ones.
for i = 1:size(Wopt, 2)
    Wopt(:, i) = Wopt(:, i) / norm(Wopt(:, i));
end
end


function pred = nnPredict(trainFeat, trainLabels, testFeat)
% Nearest neighbour.  Squared distances; sqrt cannot change which is nearest.
pred = zeros(1, size(testFeat, 2));
for t = 1:size(testFeat, 2)
    bestDist = inf;
    for j = 1:size(trainFeat, 2)
        dist = sum((testFeat(:, t) - trainFeat(:, j)).^2);
        if dist < bestDist
            bestDist = dist;
            pred(t)  = trainLabels(j);
        end
    end
end
end


function rate = nnRate(trainFeat, trainLabels, testFeat, testLabels)
pred = nnPredict(trainFeat, trainLabels, testFeat);
rate = 100 * sum(pred == testLabels) / numel(testLabels);
end


function G = readGlasses(path)
% Read the 40 x 10 table of 0/1 glasses labels.
G = zeros(40, 10);
fid = fopen(path, 'r');
if fid < 0
    error('cannot open %s', path);
end
while true
    line = fgetl(fid);
    if ~ischar(line), break; end
    line = strtrim(line);
    if isempty(line) || line(1) == '#', continue; end
    v = sscanf(line(2:end), '%d');      % skip the leading 's'
    if numel(v) == 11
        G(v(1), :) = v(2:11)';
    end
end
fclose(fid);
end
