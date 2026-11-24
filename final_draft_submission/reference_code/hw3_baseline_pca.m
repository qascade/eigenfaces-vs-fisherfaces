clc;
clear;
close all;

%% ================= PATHS =================
orlRoot  = '../ORL/';
yaleRoot = '../CroppedYale/';

figDir = 'figs/';

%% ===================================================================
%% ============================  ORL  ================================
%% ===================================================================
%% ---------- 1. LOAD IMAGES (ORL: people 1..32, 6 train / 4 test) ----------
trainImages = []; trainLabels = [];
testImages  = []; testLabels  = [];
numTrain = 6;
for p = 1:32
    folder = [orlRoot 's' num2str(p) '/'];
    files  = dir([folder '*.pgm']);
    for j = 1:length(files)
        im = double(imread([folder files(j).name]));
        if j <= numTrain
            trainImages = [trainImages, im(:)];  trainLabels = [trainLabels, p];
        else
            testImages  = [testImages, im(:)];   testLabels  = [testLabels, p];
        end
    end
end
imH = 112; imW = 92;

%% ---------- 2. PCA USING THE SMALL L MATRIX ----------
[numPixels, M] = size(trainImages);
meanFace = mean(trainImages, 2);
A = trainImages - meanFace;
L = A' * A;
[V, D] = eig(L);
[eigvals, sortIdx] = sort(diag(D), 'descend');
V = V(:, sortIdx);
eigenfaces = A * V;
for i = 1:size(eigenfaces, 2)
    eigenfaces(:, i) = eigenfaces(:, i) / norm(eigenfaces(:, i));
end
lambda = eigvals / M;                      % eigenvalues of C, needed for part (i)
testCentered = testImages - meanFace;

fprintf('\n=== ORL ===\n');
fprintf('trainImages: %d x %d, testImages: %d x %d\n', size(trainImages), size(testImages));

%% ---------- 3. RECOGNITION RATE vs k  (parts d, g) ----------
% (d)/(e)/(f) and (g) ask for two different k-lists; compute rates once
% over the (g) list and pull out the (d) subset for the (d)/(e)/(f) plots.
ks_g = [1 2 3 5 10 15 20 30 50 60 65 75 100 150 170];
ks_d = [1 2 3 5 10 15 20 30 50 75 100 150 170];
[~, d_idx] = ismember(ks_d, ks_g);

rateAll  = zeros(size(ks_g));
rateSkip = nan(size(ks_g));

for t = 1:length(ks_g)
    k = ks_g(t);
    U_k = eigenfaces(:, 1:k);
    trainProjected = U_k' * A;
    testProjected  = U_k' * testCentered;

    correct = 0;
    for i = 1:size(testProjected, 2)
        distances = sum((trainProjected - testProjected(:, i)).^2, 1);
        [~, best] = min(distances);
        if trainLabels(best) == testLabels(i)
            correct = correct + 1;
        end
    end
    rateAll(t) = 100 * correct / size(testProjected, 2);

    if k > 3
        correct = 0;
        for i = 1:size(testProjected, 2)
            distances = sum((trainProjected(4:k, :) - testProjected(4:k, i)).^2, 1);
            [~, best] = min(distances);
            if trainLabels(best) == testLabels(i)
                correct = correct + 1;
            end
        end
        rateSkip(t) = 100 * correct / size(testProjected, 2);
    end
end

% Part (d): ORL, all-coefficients only, d's k-list
figure('Visible', 'off');
plot(ks_d, rateAll(d_idx), '-o', 'LineWidth', 1.5); grid on;
xlabel('k'); ylabel('Recognition rate (%)');
title('Part (d): ORL recognition rate vs k (all eigencoefficients)');
saveas(gcf, [figDir 'q1d_orl.png']);

% Part (g): ORL, all vs except-top-3, g's k-list
figure('Visible', 'off');
plot(ks_g, rateAll, '-o', ks_g, rateSkip, '-s', 'LineWidth', 1.5); grid on;
xlabel('k'); ylabel('Recognition rate (%)');
legend('all coefficients', 'except top 3', 'Location', 'southeast');
title('Part (g): ORL, all vs excluding top-3 eigencoefficients');
saveas(gcf, [figDir 'q1g_orl.png']);

fprintf('ORL (d) rates @ ks_d: '); fprintf('%.1f ', rateAll(d_idx)); fprintf('\n');
fprintf('ORL (g) all      @ ks_g: '); fprintf('%.1f ', rateAll); fprintf('\n');
fprintf('ORL (g) skip-top3 @ ks_g: '); fprintf('%.1f ', rateSkip); fprintf('\n');

%% ---------- 4. RECONSTRUCTION OF ONE FACE (part h) ----------
x   = trainImages(:, 1);
ksR = [2 10 20 50 75 100 125 150 175];
figure('Visible', 'off');
subplot(2, 5, 1); imagesc(reshape(x, imH, imW)); colormap gray; axis image off; title('original');
for t = 1:length(ksR)
    U_k  = eigenfaces(:, 1:ksR(t));
    xRec = meanFace + U_k * (U_k' * (x - meanFace));
    subplot(2, 5, t+1); imagesc(reshape(xRec, imH, imW)); colormap gray; axis image off;
    title(['k = ' num2str(ksR(t))]);
end
saveas(gcf, [figDir 'q1h_recon.png']);

%% ---------- 5. FIRST 25 EIGENFACES (part h) ----------
figure('Visible', 'off');
for i = 1:25
    subplot(5, 5, i);
    imagesc(reshape(eigenfaces(:, i), imH, imW)); colormap gray; axis image off;
    title(['V_{' num2str(i) '}']);
end
saveas(gcf, [figDir 'q1h_eigenfaces.png']);

%% ---------- 6. PART (i) NUMERICAL CHECK ----------
fprintf('\nPart (i) check (ORL): measured error vs sum of discarded eigenvalues\n');
for k = [10 50 100]
    U_k  = eigenfaces(:, 1:k);
    Xrec = meanFace + U_k * (U_k' * A);
    err  = mean(sum((Xrec - trainImages).^2, 1));
    fprintf('  k = %3d: measured = %.4f, sum(lambda_{k+1..}) = %.4f\n', k, err, sum(lambda(k+1:end)));
end

%% ===================================================================
%% ============================  YALE  ================================
%% ===================================================================
%% ---------- 1. LOAD IMAGES ----------
% Yale folders are 'yaleB01'..'yaleB39' (14 missing), so list the
% subfolders instead of building the name from a numeric index.
d = dir(yaleRoot); d = d([d.isdir]);
names = {d.name}; names = names(~ismember(names, {'.', '..'}));
names = sort(names);                       % 'yaleB01' < 'yaleB02' < ... lexicographic = numeric here

trainImagesY = []; trainLabelsY = [];
testImagesY  = []; testLabelsY  = [];
numTrainY = 40;
for p = 1:numel(names)
    folder = [yaleRoot names{p} '/'];
    files  = dir([folder '*.pgm']);
    for j = 1:length(files)
        im = double(imread([folder files(j).name]));
        if j <= numTrainY
            trainImagesY = [trainImagesY, im(:)];  trainLabelsY = [trainLabelsY, p];
        else
            testImagesY  = [testImagesY, im(:)];   testLabelsY  = [testLabelsY, p];
        end
    end
end
imHY = 192; imWY = 168;

%% ---------- 2. PCA ----------
[numPixelsY, MY] = size(trainImagesY);
meanFaceY = mean(trainImagesY, 2);
AY = trainImagesY - meanFaceY;
LY = AY' * AY;
[VY, DY] = eig(LY);
[eigvalsY, sortIdxY] = sort(diag(DY), 'descend');
VY = VY(:, sortIdxY);
eigenfacesY = AY * VY;
for i = 1:size(eigenfacesY, 2)
    eigenfacesY(:, i) = eigenfacesY(:, i) / norm(eigenfacesY(:, i));
end
lambdaY = eigvalsY / MY;
testCenteredY = testImagesY - meanFaceY;

fprintf('\n=== Yale ===\n');
fprintf('trainImages: %d x %d, testImages: %d x %d\n', size(trainImagesY), size(testImagesY));

%% ---------- 3. RECOGNITION RATE vs k (parts e/f, g) ----------
rateAllY  = zeros(size(ks_g));
rateSkipY = nan(size(ks_g));

for t = 1:length(ks_g)
    k = ks_g(t);
    U_k = eigenfacesY(:, 1:k);
    trainProjected = U_k' * AY;
    testProjected  = U_k' * testCenteredY;

    correct = 0;
    for i = 1:size(testProjected, 2)
        distances = sum((trainProjected - testProjected(:, i)).^2, 1);
        [~, best] = min(distances);
        if trainLabelsY(best) == testLabelsY(i)
            correct = correct + 1;
        end
    end
    rateAllY(t) = 100 * correct / size(testProjected, 2);

    if k > 3
        correct = 0;
        for i = 1:size(testProjected, 2)
            distances = sum((trainProjected(4:k, :) - testProjected(4:k, i)).^2, 1);
            [~, best] = min(distances);
            if trainLabelsY(best) == testLabelsY(i)
                correct = correct + 1;
            end
        end
        rateSkipY(t) = 100 * correct / size(testProjected, 2);
    end
end

% Parts (e)/(f): Yale, all-coefficients only, d's k-list
figure('Visible', 'off');
plot(ks_d, rateAllY(d_idx), '-o', 'LineWidth', 1.5); grid on;
xlabel('k'); ylabel('Recognition rate (%)');
title('Parts (e)/(f): Yale recognition rate vs k (all eigencoefficients)');
saveas(gcf, [figDir 'q1ef_yale.png']);

% Part (g): Yale, all vs except-top-3
figure('Visible', 'off');
plot(ks_g, rateAllY, '-o', ks_g, rateSkipY, '-s', 'LineWidth', 1.5); grid on;
xlabel('k'); ylabel('Recognition rate (%)');
legend('all coefficients', 'except top 3', 'Location', 'southeast');
title('Part (g): Yale, all vs excluding top-3 eigencoefficients');
saveas(gcf, [figDir 'q1g_yale.png']);

fprintf('Yale (e/f) rates @ ks_d: '); fprintf('%.1f ', rateAllY(d_idx)); fprintf('\n');
fprintf('Yale (g) all      @ ks_g: '); fprintf('%.1f ', rateAllY); fprintf('\n');
fprintf('Yale (g) skip-top3 @ ks_g: '); fprintf('%.1f ', rateSkipY); fprintf('\n');

fprintf('\nAll figures written to %s\n', figDir);
