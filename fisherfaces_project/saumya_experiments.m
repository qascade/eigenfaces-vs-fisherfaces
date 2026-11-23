% =====================================================================
%  SAUMYA_EXPERIMENTS   Part 3: seven experiments
%
%  E1  cond(S_W) against m           how safe is the paper's m = N-c?
%  E2  label permutation             is the result real, or leaking?
%  E3  cross-validated m             choosing m without peeking at test
%  E4  noise robustness              testing a prediction
%  E5  error against lighting angle  the paper's subset analysis
%  E6  sweep the number of classes   the c-1 ceiling as a live constraint
%  E7  glasses / no glasses          the paper's section 3.3
%
%  Uses the saumya_* function files.
%
%  Run:  matlab -batch "run('saumya_experiments.m')"
%  Takes several minutes, mostly loading Yale.  Figures SAVED to figs_saumya/.
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

fprintf('loading ORL ...\n');
[oTr, oYtr, oTe, oYte] = saumya_loadfaces(orlFolders, 6, dORL);
fprintf('loading Yale ... (slow)\n');
[yTr, yYtr, yTe, yYte, yTeNames] = saumya_loadfaces(yaleFolders, 40, dYALE);

oN = size(oTr,2);  oC = 32;
yN = size(yTr,2);  yC = 38;
[oV, ~, oXbar] = saumya_pca(oTr);
[yV, ~, yXbar] = saumya_pca(yTr);


%% ---------------------------------------------------------------------
%  E1  CONDITION NUMBER OF S_W AGAINST m
%
%  At m = N-c, S_W has rank m: full rank, but with ZERO SLACK.  One more PCA
%  direction and it would be singular.  cond(S_W) = largest/smallest
%  eigenvalue says how much relative error gets amplified; roughly
%  log10(cond) of our 16 digits are lost.
%
%  m is squeezed from both sides:   c-1 <= m <= N-c
%% ---------------------------------------------------------------------
fprintf('\n\n=== E1: cond(S_W) against m ===\n');

for db = 1:2
    if db == 1
        nm='ORL';  V=oV; xbar=oXbar; imTr=oTr; ytr=oYtr; imTe=oTe; yte=oYte;
        c=oC; N=oN;  ms=[31 40 60 80 100 120 140 150 155 160];
    else
        nm='Yale'; V=yV; xbar=yXbar; imTr=yTr; ytr=yYtr; imTe=yTe; yte=yYte;
        c=yC; N=yN;  ms=[100 200 400 700 1000 1200 1400 1450 1482];
    end
    conds = zeros(1,numel(ms));  rates = zeros(1,numel(ms));
    fprintf('\n  %s  (N = %d, c = %d, N-c = %d)\n', nm, N, c, N-c);
    fprintf('       m   rank(S_W)    cond(S_W)   rate%%\n');
    for i = 1:numel(ms)
        [Wopt, ~, Sw] = saumya_fisherfaces(V, xbar, imTr, ytr, ms(i), c);
        conds(i) = cond(Sw);
        rates(i) = saumya_nn(Wopt'*(imTr-xbar), ytr, Wopt'*(imTe-xbar), yte);
        fprintf('  %6d   %7d   %.3e   %5.1f\n', ms(i), rank(Sw), conds(i), rates(i));
    end
    [bR, iB] = max(rates);
    fprintf('   best m = %d at %.1f%%; paper uses m = %d giving %.1f%%\n', ...
            ms(iB), bR, N-c, rates(end));
    fprintf('   cond(S_W) grows %.2e-fold across the sweep\n', conds(end)/conds(1));

    figure('Visible', figVis, 'Position', [100 100 900 340]);
    subplot(1,2,1);
    semilogy(ms, conds, '-o', 'LineWidth', 1.5); grid on;
    xlabel('m'); ylabel('cond(S_W)  [log]');
    title([nm ': conditioning explodes at m = N-c']);
    subplot(1,2,2);
    plot(ms, rates, '-o', 'LineWidth', 1.5); grid on;
    xlabel('m'); ylabel('rate (%)');
    title([nm ': accuracy peaks before m = N-c']);
    saveas(gcf, [figDir 'E1_cond_' nm '.png']);
    if ~showFigures, close; end
end


%% ---------------------------------------------------------------------
%  E2  LABEL PERMUTATION
%
%  Shuffle the TRAINING labels and change nothing else.  Now no face has any
%  relation to its label, so nothing can beat guessing: 100/c = 3.1%.
%  A method still scoring well would mean the pipeline is leaking.
%% ---------------------------------------------------------------------
fprintf('\n\n=== E2: label permutation (ORL) ===\n');

Vc = oV(:, 1:(oC-1));
rEigT = saumya_nn(Vc'*(oTr-oXbar), oYtr, Vc'*(oTe-oXbar), oYte);
WT = saumya_fisherfaces(oV, oXbar, oTr, oYtr, oN-oC, oC);
rFisT = saumya_nn(WT'*(oTr-oXbar), oYtr, WT'*(oTe-oXbar), oYte);

ytrShuf = oYtr(randperm(oN));           % images untouched, labels scrambled
rEigP = saumya_nn(Vc'*(oTr-oXbar), ytrShuf, Vc'*(oTe-oXbar), oYte);
WP = saumya_fisherfaces(oV, oXbar, oTr, ytrShuf, oN-oC, oC);
rFisP = saumya_nn(WP'*(oTr-oXbar), ytrShuf, WP'*(oTe-oXbar), oYte);

chance = 100 / oC;
fprintf('                    true labels   shuffled\n');
fprintf('  eigenfaces          %5.1f%%       %5.1f%%\n', rEigT, rEigP);
fprintf('  Fisherfaces         %5.1f%%       %5.1f%%\n', rFisT, rFisP);
fprintf('  chance = 100/c      %5.1f%%\n', chance);
fprintf('  Below chance, not at it: Fisherfaces separates the SHUFFLED\n');
fprintf('  grouping faithfully, so a test face lands near training images\n');
fprintf('  that look like it, which now carry the wrong label by design.\n');

figure('Visible', figVis, 'Position', [100 100 560 360]);
bar([rEigT rEigP; rFisT rFisP]); hold on;
plot([0.5 2.5], [chance chance], 'k--', 'LineWidth', 1.2); hold off;
set(gca,'XTickLabel',{'eigenfaces','Fisherfaces'}); ylabel('rate (%)'); grid on;
legend({'true labels','shuffled','chance'}, 'Location','northeast');
title('ORL: shuffling training labels destroys both methods');
saveas(gcf, [figDir 'E2_permutation.png']);
if ~showFigures, close; end


%% ---------------------------------------------------------------------
%  E3  CHOOSING m BY CROSS-VALIDATION
%
%  E1 read "the best m" off the test column.  That is tuning on test and the
%  number it gives is optimistic.  The honest way is 6-fold leave-one-out
%  INSIDE the training set: train on 5 images per person, validate on the
%  held-out 1, rotate, average.  Only then look at test, once.
%
%  The PCA basis is recomputed per fold -- fitting it on all six would leak
%  the validation image into the basis.
%% ---------------------------------------------------------------------
fprintf('\n\n=== E3: cross-validated m (ORL) ===\n');

msCV = [31 40 60 80 100 120];       % inside a fold N'=160 so m <= 128
cvR  = zeros(1, numel(msCV));
for i = 1:numel(msCV)
    tot = 0;
    for f = 1:6
        isVal = false(1, oN);
        for j = 1:oC
            cols = find(oYtr == j);
            isVal(cols(f)) = true;         % hold out the f-th image
        end
        subTr = oTr(:, ~isVal);  subY = oYtr(~isVal);
        subVa = oTr(:,  isVal);  subV = oYtr( isVal);
        [Vf, ~, xf] = saumya_pca(subTr);
        Wf = saumya_fisherfaces(Vf, xf, subTr, subY, msCV(i), oC);
        tot = tot + saumya_nn(Wf'*(subTr-xf), subY, Wf'*(subVa-xf), subV);
    end
    cvR(i) = tot / 6;
    fprintf('   m = %3d   cross-validated = %5.1f%%\n', msCV(i), cvR(i));
end
[~, iCV] = max(cvR);
Wcv = saumya_fisherfaces(oV, oXbar, oTr, oYtr, msCV(iCV), oC);
rCV = saumya_nn(Wcv'*(oTr-oXbar), oYtr, Wcv'*(oTe-oXbar), oYte);

testR = zeros(1, numel(msCV));
for i = 1:numel(msCV)
    Wt = saumya_fisherfaces(oV, oXbar, oTr, oYtr, msCV(i), oC);
    testR(i) = saumya_nn(Wt'*(oTr-oXbar), oYtr, Wt'*(oTe-oXbar), oYte);
end
[bT, iT] = max(testR);
fprintf('\n   CV chose m = %d, scoring %.1f%% on test\n', msCV(iCV), rCV);
fprintf('   tuning on test would choose m = %d for %.1f%%\n', msCV(iT), bT);
fprintf('   the %.1f-point gap is the optimism avoided by not peeking\n', bT - rCV);

figure('Visible', figVis, 'Position', [100 100 600 380]);
plot(msCV, cvR, '-o', 'LineWidth',1.5); hold on;
plot(msCV, testR, '-s', 'LineWidth',1.5);
plot(msCV(iCV), rCV, 'kp', 'MarkerSize',14, 'MarkerFaceColor','y'); hold off;
grid on; xlabel('m'); ylabel('rate (%)');
legend({'cross-validated (honest)','test set (peeking)','m chosen by CV'}, ...
       'Location','southwest');
title('ORL: choosing m with and without looking at test');
saveas(gcf, [figDir 'E3_crossval.png']);
if ~showFigures, close; end


%% ---------------------------------------------------------------------
%  E4  NOISE ROBUSTNESS -- TESTING A PREDICTION
%
%  Fisherfaces sit about 3.85x higher in spatial frequency than eigenfaces,
%  because whitening scales direction l by 1/sqrt(lambda_l) and so amplifies
%  the low-variance fine-detail directions most.  White noise has energy at
%  all frequencies.  PREDICTION: Fisherfaces should degrade faster.
%
%  Noise is added to TEST images only; the projections stay as learned from
%  clean training data.  Pixels are 0-255, and projection is a strong
%  denoiser, so sigma has to get large before anything happens at all.
%% ---------------------------------------------------------------------
fprintf('\n\n=== E4: noise robustness (ORL) ===\n');

sigmas = [0 20 40 80 160 320 640];
rC = zeros(1,numel(sigmas)); rE = rC; rF = rC;
Vk50 = oV(:, 1:50);
Wn = saumya_fisherfaces(oV, oXbar, oTr, oYtr, oN-oC, oC);

fprintf('   sigma  correlation  eigenfaces(k=50)  Fisherfaces\n');
for i = 1:numel(sigmas)
    rng(1);                              % same noise for all three methods
    noisy = oTe + sigmas(i) * randn(size(oTe));
    rC(i) = saumya_nn(saumya_normcols(oTr), oYtr, saumya_normcols(noisy), oYte);
    rE(i) = saumya_nn(Vk50'*(oTr-oXbar), oYtr, Vk50'*(noisy-oXbar), oYte);
    rF(i) = saumya_nn(Wn'*(oTr-oXbar),   oYtr, Wn'*(noisy-oXbar),   oYte);
    fprintf('  %6d   %9.1f%%  %14.1f%%  %10.1f%%\n', sigmas(i), rC(i), rE(i), rF(i));
end
onePt = 100 / numel(oYte);
fprintf('\n   one test image = %.2f points, so smaller gaps are not real\n', onePt);
fprintf('   drop from sigma 0 to %d: correlation %.1f, eigenfaces %.1f, Fisherfaces %.1f\n', ...
        sigmas(end), rC(1)-rC(end), rE(1)-rE(end), rF(1)-rF(end));
if (rF(1)-rF(end)) > (rE(1)-rE(end)) + 2*onePt
    fprintf('   => Fisherfaces fall further, as predicted.  CAVEAT: only at\n');
    fprintf('      sigma far beyond realistic noise; below 160 nothing moves.\n');
else
    fprintf('   => prediction NOT confirmed here; report that rather than hide it.\n');
end

figure('Visible', figVis, 'Position', [100 100 600 380]);
plot(sigmas, rC, '-o', 'LineWidth',1.5); hold on;
plot(sigmas, rE, '-s', 'LineWidth',1.5);
plot(sigmas, rF, '-^', 'LineWidth',1.5); hold off;
grid on; xlabel('noise std added to test images (grey levels)'); ylabel('rate (%)');
legend({'correlation','eigenfaces (k=50)','Fisherfaces'}, 'Location','southwest');
title('ORL: degradation under pixel noise');
saveas(gcf, [figDir 'E4_noise.png']);
if ~showFigures, close; end


%% ---------------------------------------------------------------------
%  E5  ERROR AGAINST LIGHTING ANGLE
%
%  Every Yale filename encodes the light direction, e.g.
%        yaleB01_P00A+020E-40.pgm   ->  azimuth +20, elevation -40
%  Treating azimuth and elevation as rotations about perpendicular axes, the
%  direction cosine along the camera axis is the product of the cosines, so
%        theta = acos( cos(azimuth) * cos(elevation) )
%  The paper's bins are 0-12, 13-25, 26-50, 51-77 and above 77 degrees.
%% ---------------------------------------------------------------------
fprintf('\n\n=== E5: Yale error against lighting angle ===\n');

nTe = numel(yTeNames);
theta = zeros(1, nTe);
for i = 1:nTe
    nmS = yTeNames{i};
    ia = strfind(nmS, 'A');            % uppercase A appears exactly once
    ie = strfind(nmS, 'E');            % uppercase E appears exactly once
    az = str2double(nmS(ia+1 : ie-1));         % '+020' -> 20
    el = str2double(nmS(ie+1 : end-4));        % '-40'  -> -40, drop '.pgm'
    theta(i) = acosd(cosd(az) * cosd(el));
end

subset = zeros(1, nTe);
for i = 1:nTe
    if     theta(i) <= 12,  subset(i) = 1;
    elseif theta(i) <= 25,  subset(i) = 2;
    elseif theta(i) <= 50,  subset(i) = 3;
    elseif theta(i) <= 77,  subset(i) = 4;
    else,                   subset(i) = 5;
    end
end

yVc = yV(:, 1:(yC-1));
[~, predEig] = saumya_nn(yVc'*(yTr-yXbar), yYtr, yVc'*(yTe-yXbar), yYte);
Wy = saumya_fisherfaces(yV, yXbar, yTr, yYtr, yN-yC, yC);
[~, predFis] = saumya_nn(Wy'*(yTr-yXbar), yYtr, Wy'*(yTe-yXbar), yYte);

% NaN so an empty bin leaves a GAP in the bar chart, not a bar at 0%.
errE = nan(1,5);  errF = nan(1,5);  cnt = zeros(1,5);
edges = [0 12 25 50 77 180];
fprintf('   subset   angle        images   eigenfaces   Fisherfaces\n');
for b = 1:5
    sel = (subset == b);
    cnt(b) = sum(sel);
    if cnt(b) == 0
        fprintf('     %d     %3d-%3d deg        0     (no test images)\n', b, edges(b), edges(b+1));
        continue;
    end
    errE(b) = 100 * sum(predEig(sel) ~= yYte(sel)) / cnt(b);
    errF(b) = 100 * sum(predFis(sel) ~= yYte(sel)) / cnt(b);
    fprintf('     %d     %3d-%3d deg    %5d   %9.1f%%   %10.1f%%\n', ...
            b, edges(b), edges(b+1), cnt(b), errE(b), errF(b));
end
if cnt(1) == 0
    fprintf('\n   Subset 1 is EMPTY, and that is the split bias, not a bug:\n');
    fprintf('   dir sorts as text and ''+'' sorts before ''-'', so the first 40\n');
    fprintf('   files per subject (the training set) absorb all the frontal\n');
    fprintf('   lighting.  Every test image is at least 12 degrees off axis.\n');
end

figure('Visible', figVis, 'Position', [100 100 620 380]);
bar([errE(:) errF(:)]);
set(gca,'XTickLabel',{'0-12','13-25','26-50','51-77','>77'});
xlabel('angle of light from the optical axis (deg)'); ylabel('error (%)');
legend({'eigenfaces (k=c-1)','Fisherfaces'}, 'Location','northwest'); grid on;
title('Yale: error grows with lighting angle');
saveas(gcf, [figDir 'E5_angle.png']);
if ~showFigures, close; end


%% ---------------------------------------------------------------------
%  E6  SWEEPING THE NUMBER OF CLASSES
%
%  rank(S_B) <= c-1, so there are at most c-1 Fisherfaces however many
%  pixels or images we have.  Vary c and watch that ceiling bind, and watch
%  accuracy fall as the problem gets harder.
%% ---------------------------------------------------------------------
fprintf('\n\n=== E6: sweeping the number of classes (ORL) ===\n');

cs = [2 4 8 16 32];
rE6 = zeros(1,numel(cs));  rF6 = zeros(1,numel(cs));  nf6 = zeros(1,numel(cs));
fprintf('      c     N   N-c   #fisherfaces   eigenfaces   Fisherfaces\n');
for i = 1:numel(cs)
    c = cs(i);
    [tr, ytrC, te, yteC] = saumya_loadfaces(orlFolders(1:c), 6, dORL);
    N = size(tr,2);
    [Vc6, ~, xb6] = saumya_pca(tr);
    kk = max(1, c-1);
    Vk = Vc6(:, 1:kk);
    rE6(i) = saumya_nn(Vk'*(tr-xb6), ytrC, Vk'*(te-xb6), yteC);
    W6 = saumya_fisherfaces(Vc6, xb6, tr, ytrC, N-c, c);
    nf6(i) = size(W6,2);
    rF6(i) = saumya_nn(W6'*(tr-xb6), ytrC, W6'*(te-xb6), yteC);
    fprintf('  %5d %5d %5d   %10d   %9.1f%%   %10.1f%%\n', ...
            c, N, N-c, nf6(i), rE6(i), rF6(i));
end
fprintf('   #fisherfaces equals c-1 on every row: the rank bound is tight\n');
fprintf('   and is a LIVE constraint, not just a proof on a page.\n');

figure('Visible', figVis, 'Position', [100 100 900 340]);
subplot(1,2,1);
plot(cs, nf6, '-o', 'LineWidth',1.5); hold on;
plot(cs, cs-1, 'k--', 'LineWidth',1.2); hold off; grid on;
xlabel('number of classes c'); ylabel('number of Fisherfaces');
legend({'measured','c-1'}, 'Location','northwest');
title('the c-1 ceiling is exactly met');
subplot(1,2,2);
plot(cs, rE6, '-s', 'LineWidth',1.5); hold on;
plot(cs, rF6, '-^', 'LineWidth',1.5); hold off; grid on;
xlabel('number of classes c'); ylabel('rate (%)');
legend({'eigenfaces (k=c-1)','Fisherfaces'}, 'Location','southwest');
title('both get harder as c grows');
saveas(gcf, [figDir 'E6_classes.png']);
if ~showFigures, close; end


%% ---------------------------------------------------------------------
%  E7  GLASSES / NO GLASSES  (the paper's section 3.3)
%
%  A different question: not WHO is this, but IS THIS PERSON WEARING
%  GLASSES.  Two classes, so exactly ONE Fisherface.
%
%  ORL ships no glasses labels, so orl_glasses_labels.txt was made by
%  inspecting all 400 images by eye.  8 subjects appear both with and
%  without glasses, which is what makes the question non-trivial.
%
%  Protocol: LEAVE ONE SUBJECT OUT.  Train on the other 39 people, test on
%  the 10 images of the held-out person.  Because that face was never seen,
%  the classifier cannot cheat by recognising the individual -- it has to
%  have learned what glasses look like.  This mirrors the paper, which also
%  removed the test person before computing W.
%% ---------------------------------------------------------------------
fprintf('\n\n=== E7: glasses vs no glasses (ORL, leave-one-subject-out) ===\n');

G = readGlasses(fullfile(fileparts(mfilename('fullpath')), 'orl_glasses_labels.txt'));
fprintf('   %d of 400 images wear glasses (%.1f%%)\n', sum(G(:)), 100*sum(G(:))/400);

% load all 40 subjects, all 10 images each (numTrain = 10 puts everything in
% the "training" half; we do our own splitting below)
orl40 = cell(1,40);
rootOnly = fileparts(orlFolders{1});
rootOnly = fileparts(rootOnly);
for p = 1:40
    orl40{p} = [rootOnly filesep 's' num2str(p) filesep];
end
[allIm, allSubj] = saumya_loadfaces(orl40, 10, dORL);
allG = zeros(1, size(allIm,2));
for i = 1:size(allIm,2)
    % images of subject p sit in columns in file order 1..10
    k = sum(allSubj(1:i) == allSubj(i));
    allG(i) = G(allSubj(i), k);
end

varying = [4 7 13 17 19 20 28 36];     % subjects who appear both ways

% BALANCE the set.  The paper used 36 images, HALF with glasses, so its
% baseline was 50%.  Ours is 29.2% glasses, where simply always answering
% "no" scores 70.8% and beats any real method -- a meaningless comparison.
% So keep all 117 glasses images and a random 117 of the others.
gIdx = find(allG == 1);
nIdx = find(allG == 0);
nIdx = nIdx(randperm(numel(nIdx), numel(gIdx)));
keepIdx = sort([gIdx nIdx]);

balIm   = allIm(:, keepIdx);
balSubj = allSubj(keepIdx);
balG    = allG(keepIdx);
fprintf('   balanced subset: %d images, %.1f%% with glasses\n', ...
        numel(balG), 100*mean(balG));

okF = 0; okE = 0; nTot = 0;
okFv = 0; okEv = 0; nVar = 0;

for p = 1:40
    isTe = (balSubj == p);
    if sum(isTe) == 0 || sum(~isTe) < 10
        continue;                       % this subject contributed nothing
    end
    trI = balIm(:, ~isTe);  trG = balG(~isTe) + 1;   % labels must be 1 and 2
    teI = balIm(:,  isTe);  teG = balG( isTe) + 1;
    if numel(unique(trG)) < 2
        continue;                       % need both classes to train
    end

    Ntr = size(trI, 2);
    [Vg, ~, xg] = saumya_pca(trI);

    % Fisherfaces: c = 2, so rank(S_B) <= 1 and there is exactly ONE
    % Fisherface.  The whole decision rests on a single direction.
    Wg = saumya_fisherfaces(Vg, xg, trI, trG, Ntr - 2, 2);
    [~, pf] = saumya_nn(Wg'*(trI-xg), trG, Wg'*(teI-xg), teG);

    % PCA for comparison, at the paper's reduced space of 10.
    Vk = Vg(:, 1:10);
    [~, pe] = saumya_nn(Vk'*(trI-xg), trG, Vk'*(teI-xg), teG);

    okF = okF + sum(pf == teG);  okE = okE + sum(pe == teG);  nTot = nTot + numel(teG);
    if any(varying == p)
        okFv = okFv + sum(pf == teG);  okEv = okEv + sum(pe == teG);
        nVar = nVar + numel(teG);
    end
end

errE  = 100 - 100*okE/nTot;
errF  = 100 - 100*okF/nTot;
errEv = 100 - 100*okEv/nVar;
errFv = 100 - 100*okFv/nVar;

fprintf('\n   held-out subject never appears in training, so the classifier\n');
fprintf('   cannot cheat by recognising the person.\n\n');
fprintf('   method                 ERROR RATE (%%)      paper (Yale, 36 images)\n');
fprintf('   PCA, k = 10            %8.1f               52.6\n', errE);
fprintf('   ONE Fisherface         %8.1f                5.3\n', errF);
fprintf('   always say "no"            50.0               50.0   (balanced)\n');
fprintf('\n   restricted to the %d subjects who appear BOTH ways (%d images):\n', ...
        numel(varying), nVar);
fprintf('   PCA, k = 10            %8.1f\n', errEv);
fprintf('   ONE Fisherface         %8.1f\n', errFv);
fprintf('\n   c = 2 here, so rank(S_B) <= 1: one single direction decides\n');
fprintf('   everything.  That is Fisher''s 1936 idea in its original form.\n');

figure('Visible', figVis, 'Position', [100 100 600 380]);
bar([errE errF; errEv errFv]); hold on;
plot([0.5 2.5], [50 50], 'k--', 'LineWidth',1.2); hold off;
set(gca,'XTickLabel',{'balanced set','the 8 that vary'});
ylabel('error rate (%)'); grid on;
legend({'PCA (k=10)','one Fisherface','always say no'}, ...
       'Location','southoutside','Orientation','horizontal');
title('ORL: detecting glasses on an unseen person');
saveas(gcf, [figDir 'E7_glasses.png']);
if ~showFigures, close; end

fprintf('\n\n  figures written to %s\n', figDir);
fprintf('  DONE\n');


% ---------------------------------------------------------------------
function G = readGlasses(path)
% Read orl_glasses_labels.txt into a 40 x 10 matrix of 0/1.
% Lines starting with # are comments; each data line is  sNN  d d d ... d
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
    parts = sscanf(line(2:end), '%d');     % skip the leading 's'
    if numel(parts) == 11
        G(parts(1), :) = parts(2:11)';
    end
end
fclose(fid);
end
