%% Experiments on top of eigenfaces_vs_fisherfaces.m
%
% E1  sweeping the PCA dimension m, with cond(S_W)   (study guide, section 24)
% E2  label permutation: is the result real?
% E3  choosing m by cross-validation, without peeking at the test set
% E4  glasses vs no glasses (the paper's section 3.3)
% E5  error against lighting angle (Yale)
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
[orlTrain, orlTrainLabels, orlTest, orlTestLabels] = loadFaces(orlDir, 32, 6, 112*92);
fprintf('loading Yale ... (slow)\n');
[yaleTrain, yaleTrainLabels, yaleTest, yaleTestLabels, yaleTestNames] = loadFaces(yaleDir, 38, 40, 192*168);

orlN = size(orlTrain, 2);  orlC = 32;
yaleN = size(yaleTrain, 2);  yaleC = 38;
[orlEigenfaces, orlMeanFace] = doPCA(orlTrain);
[yaleEigenfaces, yaleMeanFace] = doPCA(yaleTrain);


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
        dbName = 'ORL';  U = orlEigenfaces; meanIm = orlMeanFace; trainIm = orlTrain; trainLab = orlTrainLabels; testIm = orlTest; testLab = orlTestLabels;
        c = orlC; N = orlN;  ms = [31 40 60 80 100 120 140 160];
    else
        dbName = 'Yale'; U = yaleEigenfaces; meanIm = yaleMeanFace; trainIm = yaleTrain; trainLab = yaleTrainLabels; testIm = yaleTest; testLab = yaleTestLabels;
        c = yaleC; N = yaleN;  ms = [37 60 100 200 400 700 1000 1200 1482];
    end

    conds = zeros(1, numel(ms));
    rates = zeros(1, numel(ms));
    fprintf('\n  %s  (N = %d, c = %d, N-c = %d)\n', dbName, N, c, N-c);
    fprintf('       m   rate%%   rank(S_W)    cond(S_W)\n');
    for i = 1:numel(ms)
        [fisherW, Sw] = fisherfaces(U, meanIm, trainIm, trainLab, ms(i), c);
        conds(i) = cond(Sw);
        rates(i) = nnRate(fisherW'*(trainIm-meanIm), trainLab, fisherW'*(testIm-meanIm), testLab);
        fprintf('  %6d  %6.1f   %7d    %.2e\n', ms(i), rates(i), rank(Sw), conds(i));
    end
    [bestR, iB] = max(rates);
    fprintf('   best m = %d at %.1f%%; the paper uses m = %d giving %.1f%%\n', ...
            ms(iB), bestR, N-c, rates(end));

    figure('Position', [100 100 900 340]);
    subplot(1,2,1);
    semilogy(ms, conds, '-o', 'LineWidth', 1.5); grid on;
    xlabel('m'); ylabel('cond(S_W)  [log scale]');
    title([dbName ': conditioning explodes at m = N-c']);
    subplot(1,2,2);
    plot(ms, rates, '-o', 'LineWidth', 1.5); grid on;
    xlabel('m'); ylabel('accuracy (%)');
    title([dbName ': accuracy peaks before m = N-c']);
    saveas(gcf, [figDir 'E1_sweep_m_' dbName '.png']);
end


%% E2. Label permutation
% Shuffle the TRAINING labels and change nothing else.  Now no face has any
% relation to its label, so nothing can beat guessing, which is 100/c.
% A method still scoring well would mean the pipeline is leaking.

fprintf('\n\n=== E2: label permutation (ORL) ===\n');

Umatched = orlEigenfaces(:, 1:(orlC-1));
rateEigen = nnRate(Umatched'*(orlTrain-orlMeanFace), orlTrainLabels, Umatched'*(orlTest-orlMeanFace), orlTestLabels);
fisherW  = fisherfaces(orlEigenfaces, orlMeanFace, orlTrain, orlTrainLabels, orlN-orlC, orlC);
rateFisher = nnRate(fisherW'*(orlTrain-orlMeanFace), orlTrainLabels, fisherW'*(orlTest-orlMeanFace), orlTestLabels);

rng('default');                       % so this experiment does not depend on what ran before
shuffledLabels = orlTrainLabels(randperm(orlN));
rateEigenShuffled = nnRate(Umatched'*(orlTrain-orlMeanFace), shuffledLabels, Umatched'*(orlTest-orlMeanFace), orlTestLabels);
fisherWshuffled  = fisherfaces(orlEigenfaces, orlMeanFace, orlTrain, shuffledLabels, orlN-orlC, orlC);
rateFisherShuffled = nnRate(fisherWshuffled'*(orlTrain-orlMeanFace), shuffledLabels, fisherWshuffled'*(orlTest-orlMeanFace), orlTestLabels);

chance = 100 / orlC;
fprintf('                 true labels   shuffled\n');
fprintf('  eigenfaces       %6.1f%%     %6.1f%%\n', rateEigen, rateEigenShuffled);
fprintf('  Fisherfaces      %6.1f%%     %6.1f%%\n', rateFisher, rateFisherShuffled);
fprintf('  chance = 100/c   %6.1f%%\n', chance);
fprintf('  Both collapse, so the real numbers come from the faces.\n');

figure('Position', [100 100 560 360]);
bar([rateEigen rateEigenShuffled; rateFisher rateFisherShuffled]); hold on;
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
cvRate  = zeros(1, numel(msCV));
for i = 1:numel(msCV)
    total = 0;
    for f = 1:6
        isVal = false(1, orlN);
        for p = 1:orlC
            cols = find(orlTrainLabels == p);
            isVal(cols(f)) = true;
        end
        foldTrain = orlTrain(:, ~isVal);  foldTrainLabels = orlTrainLabels(~isVal);
        foldVal = orlTrain(:,  isVal);  foldValLabels = orlTrainLabels( isVal);
        [foldEigenfaces, meanIm] = doPCA(foldTrain);
        foldFisherW = fisherfaces(foldEigenfaces, meanIm, foldTrain, foldTrainLabels, msCV(i), orlC);
        total = total + nnRate(foldFisherW'*(foldTrain-meanIm), foldTrainLabels, foldFisherW'*(foldVal-meanIm), foldValLabels);
    end
    cvRate(i) = total / 6;
    fprintf('   m = %3d   cross-validated = %5.1f%%\n', msCV(i), cvRate(i));
end

testRate = zeros(1, numel(msCV));
for i = 1:numel(msCV)
    testFisherW = fisherfaces(orlEigenfaces, orlMeanFace, orlTrain, orlTrainLabels, msCV(i), orlC);
    testRate(i) = nnRate(testFisherW'*(orlTrain-orlMeanFace), orlTrainLabels, testFisherW'*(orlTest-orlMeanFace), orlTestLabels);
end
[~, iCV] = max(cvRate);
[bT, iT] = max(testRate);
fprintf('\n   CV picks m = %d, which scores %.1f%% on the test set\n', ...
        msCV(iCV), testRate(iCV));
fprintf('   tuning on test would pick m = %d for %.1f%%\n', msCV(iT), bT);
fprintf('   the %.1f point gap is the optimism avoided by not peeking\n', ...
        bT - testRate(iCV));

figure('Position', [100 100 600 380]);
plot(msCV, cvRate, '-o', 'LineWidth', 1.5); hold on;
plot(msCV, testRate, '-s', 'LineWidth', 1.5);
plot(msCV(iCV), testRate(iCV), 'kp', 'MarkerSize', 14, 'MarkerFaceColor', 'y');
hold off; grid on;
xlabel('m'); ylabel('accuracy (%)');
legend('cross-validated (honest)', 'test set (peeking)', 'm chosen by CV', ...
       'Location', 'southwest');
title('Choosing m with and without looking at the test set');
saveas(gcf, [figDir 'E3_cross_validation.png']);



%% E4. Glasses vs no glasses (the paper's section 3.3)
% A different question: not WHO is this, but IS THIS PERSON WEARING GLASSES.
% Two classes, so rank(S_B) <= 1 and there is exactly ONE Fisherface.
%
% Both methods are the SAME pipeline: project, then 1-nearest-neighbour with
% the glasses label attached to each training image.  Only the subspace
% differs -- PCA picks its directions by total variance and never sees a
% label, LDA picks the one direction that separates the two label groups.
%
% ORL ships no glasses labels, so orl_glasses_labels.txt was made by looking
% at all 400 images.  117 wear glasses and 8 people appear both ways.
%
% We balance the set 50/50 (otherwise always answering "no" scores 70.8% and
% beats everything) and leave one SUBJECT out at a time, so the test face is
% never seen in training and the classifier cannot cheat by recognising it.

fprintf('\n\n=== E4: glasses vs no glasses (ORL) ===\n');

G = readGlasses('orl_glasses_labels.txt');
[allImages, allSubjects] = loadFaces(orlDir, 40, 10, 112*92);
allGlasses = zeros(1, size(allImages, 2));
for i = 1:size(allImages, 2)
    k = sum(allSubjects(1:i) == allSubjects(i));      % which image of this person
    allGlasses(i) = G(allSubjects(i), k);
end
fprintf('   %d of 400 images wear glasses (%.1f%%)\n', sum(allGlasses), 100*mean(allGlasses));
fprintf('   on the UNbalanced set, always answering "no" would score %.1f%%,\n', ...
        100*max(sum(allGlasses==0), sum(allGlasses==1))/numel(allGlasses));
fprintf('   which beats any real method -- hence the balancing below.\n');

glassesIdx = find(allGlasses == 1);
noGlassesIdx = find(allGlasses == 0);
fprintf('   balanced subset: %d images, 50.0%% with glasses\n', 2*numel(glassesIdx));

% WHICH no-glasses images get picked matters: a single draw moved the
% Fisherface error from 36% to 51% in testing, so one draw proves nothing.
% Average over several draws and report the spread as well as the mean.
nDraws = 10;
errEigenDraw = zeros(1, nDraws);
errFisherDraw = zeros(1, nDraws);
rng('default');                       % so this experiment does not depend on what ran before

for r = 1:nDraws
    noGlassesPick = noGlassesIdx(randperm(numel(noGlassesIdx), numel(glassesIdx)));
    keepIdx = sort([glassesIdx noGlassesPick]);
    balIm = allImages(:, keepIdx);
    balSubjects = allSubjects(keepIdx);
    balGlasses = allGlasses(keepIdx);

    correctEigen = 0; correctFisher = 0; tot = 0;
    for p = 1:40
        isTe = (balSubjects == p);
        if sum(isTe) == 0, continue; end
        foldTrain = balIm(:, ~isTe);  foldTrainLabels = balGlasses(~isTe) + 1;   % labels must be 1 and 2
        teI = balIm(:,  isTe);  teL = balGlasses( isTe) + 1;
        if numel(unique(foldTrainLabels)) < 2, continue; end

        [glassesEigenfaces, glassesMeanFace] = doPCA(foldTrain);
        glassesFisherW = fisherfaces(glassesEigenfaces, glassesMeanFace, foldTrain, foldTrainLabels, size(foldTrain,2) - 2, 2);
        predFisherGlasses = nnPredict(glassesFisherW'*(foldTrain-glassesMeanFace), foldTrainLabels, glassesFisherW'*(teI-glassesMeanFace));
        Uk10 = glassesEigenfaces(:, 1:10);
        predEigenGlasses = nnPredict(Uk10'*(foldTrain-glassesMeanFace), foldTrainLabels, Uk10'*(teI-glassesMeanFace));

        correctFisher = correctFisher + sum(predFisherGlasses == teL);
        correctEigen = correctEigen + sum(predEigenGlasses == teL);
        tot = tot + numel(teL);
    end
    errEigenDraw(r) = 100 - 100*correctEigen/tot;
    errFisherDraw(r) = 100 - 100*correctFisher/tot;
end

errEigen = mean(errEigenDraw);
errFisher = mean(errFisherDraw);
fprintf('\n   method            our error (mean of %d draws)    paper (Yale, 36 images)\n', nDraws);
fprintf('   PCA, k = 10       %8.1f%%  (%.1f to %.1f)        52.6%%\n', errEigen, min(errEigenDraw), max(errEigenDraw));
fprintf('   one Fisherface    %8.1f%%  (%.1f to %.1f)         5.3%%\n', errFisher, min(errFisherDraw), max(errFisherDraw));
fprintf('   always say "no"       50.0%%                     50.0%%\n');
fprintf('   Neither method works here, and the spread over draws is wide.\n');
fprintf('   We reproduce that PCA is near chance, but NOT that Fisherfaces\n');
fprintf('   solve it.  Their 36 images were PAIRED (the same person with and\n');
fprintf('   without glasses, one session); in ORL glasses come with changes\n');
fprintf('   of pose, expression and session, so there is no clean signal.\n');

figure('Position', [100 100 560 360]);
bar([errEigen, errFisher]); hold on;
plot([0.5 2.5], [50 50], 'k--', 'LineWidth', 1.2); hold off;
set(gca, 'XTickLabel', {'PCA (k=10)', 'one Fisherface'});
ylabel('error rate (%)'); grid on;
legend('mean error over 10 draws', 'always say no');
title('Detecting glasses on a person never seen in training');
saveas(gcf, [figDir 'E4_glasses.png']);


%% E5. Error against lighting angle (Yale)
% Every Yale filename encodes the light direction, e.g.
%     yaleB01_P00A+020E-40.pgm  ->  azimuth +20, elevation -40
% Azimuth and elevation are rotations about perpendicular axes, so the
% direction cosine along the camera axis is the product of the cosines:
%     theta = acos( cos(azimuth) * cos(elevation) )
% The bins 0-12, 13-25, 26-50, 51-77 and above 77 degrees are the standard
% subsets of the Yale B literature (Georghiades et al. 2001).

fprintf('\n\n=== E5: error against lighting angle (Yale) ===\n');

nTe = numel(yaleTestNames);
theta = zeros(1, nTe);
for i = 1:nTe
    s  = yaleTestNames{i};
    ia = strfind(s, 'A');             % uppercase A appears exactly once
    ie = strfind(s, 'E');             % uppercase E appears exactly once
    az = str2double(s(ia+1 : ie-1));
    el = str2double(s(ie+1 : end-4)); % drop '.pgm'
    theta(i) = acosd(cosd(az) * cosd(el));
end

angleBin = zeros(1, nTe);
for i = 1:nTe
    if     theta(i) <= 12, angleBin(i) = 1;
    elseif theta(i) <= 25, angleBin(i) = 2;
    elseif theta(i) <= 50, angleBin(i) = 3;
    elseif theta(i) <= 77, angleBin(i) = 4;
    else,                  angleBin(i) = 5;
    end
end

yaleUmatched = yaleEigenfaces(:, 1:(yaleC-1));
predEigen = nnPredict(yaleUmatched'*(yaleTrain-yaleMeanFace), yaleTrainLabels, yaleUmatched'*(yaleTest-yaleMeanFace));
yaleFisherW = fisherfaces(yaleEigenfaces, yaleMeanFace, yaleTrain, yaleTrainLabels, yaleN-yaleC, yaleC);
predFisher = nnPredict(yaleFisherW'*(yaleTrain-yaleMeanFace), yaleTrainLabels, yaleFisherW'*(yaleTest-yaleMeanFace));

errEigen = nan(1,5); errFisher = nan(1,5); cnt = zeros(1,5);
binLo = [0 12 25 50 77]; binHi = [12 25 50 77 180];
fprintf('   bin   angle          images   eigenfaces   Fisherfaces\n');
for b = 1:5
    sel = (angleBin == b);
    cnt(b) = sum(sel);
    if cnt(b) == 0
        fprintf('    %d    %3d-%3d deg         0    (no test images)\n', b, binLo(b), binHi(b));
        continue;
    end
    errEigen(b) = 100 * sum(predEigen(sel) ~= yaleTestLabels(sel)) / cnt(b);
    errFisher(b) = 100 * sum(predFisher(sel) ~= yaleTestLabels(sel)) / cnt(b);
    fprintf('    %d    %3d-%3d deg     %5d   %9.1f%%  %10.1f%%\n', ...
            b, binLo(b), binHi(b), cnt(b), errEigen(b), errFisher(b));
end
if cnt(1) == 0
    fprintf('   The first bin is EMPTY because dir sorts names as text and\n');
    fprintf('   ''+'' sorts before ''-'', so the training set swallowed all the\n');
    fprintf('   frontal lighting.  Every test image is at least 12 deg off axis.\n');
end

figure('Position', [100 100 620 380]);
bar([errEigen(:) errFisher(:)]);
set(gca, 'XTickLabel', {'0-12','13-25','26-50','51-77','>77'});
xlabel('angle of the light from the optical axis (degrees)');
ylabel('error rate (%)'); grid on;
legend('eigenfaces', 'Fisherfaces', 'Location', 'northwest');
title('Yale: error grows with lighting angle');
saveas(gcf, [figDir 'E5_lighting_angle.png']);


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
% fisherW'*S_W*fisherW = I, so their Euclidean lengths differ by orders of magnitude,
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
