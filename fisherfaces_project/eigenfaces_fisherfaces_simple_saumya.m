%% Eigenfaces and Fisherfaces: simple implementation
% Only basic MATLAB: loops, matrix multiplication, mean, eig, norm.
%
% Folder layout expected (like ORL):
%   dataDir/s1/1.pgm, dataDir/s1/2.pgm, ...
%   dataDir/s2/1.pgm, ...
% One subfolder per person, all images the same size.

clc; clear; close all;

dataDir   = 'ORL';      % folder with one subfolder per person
numPeople = 40;         % c = number of people (classes)
numTrain  = 5;          % training images per person (rest are test)
k         = 30;         % number of eigenfaces to keep

%% 1. Load images and split into training and testing
trainImages = []; trainLabels = [];
testImages  = []; testLabels  = [];

for p = 1:numPeople
    folder = [dataDir '/s' num2str(p) '/'];
    files  = dir([folder '*.pgm']);
    for j = 1:length(files)
        im = double(imread([folder files(j).name]));
        if j <= numTrain
            trainImages = [trainImages, im(:)];   % each image = one column
            trainLabels = [trainLabels, p];
        else
            testImages  = [testImages, im(:)];
            testLabels  = [testLabels, p];
        end
    end
end

[h, w] = size(im);                 % image height and width (for display)
n = size(trainImages, 1);          % number of pixels
N = size(trainImages, 2);          % number of training images
c = numPeople;                     % number of classes

%% 2. Subtract the mean face
meanFace = mean(trainImages, 2);   % n x 1
A = trainImages - meanFace;        % n x N, centred training images

%% 3. PCA using the small matrix L = A'*A (N x N instead of n x n)
L = A' * A;                        % N x N
[V, D] = eig(L);                   % eigenvectors of L
[vals, order] = sort(diag(D), 'descend');
V = V(:, order);                   % largest eigenvalue first

% If L*v = lambda*v, then (A*A')*(A*v) = lambda*(A*v),
% so A*v is an eigenvector of the big covariance matrix.
U = A * V;                         % n x N
for i = 1:N
    U(:, i) = U(:, i) / norm(U(:, i));   % make each one unit length
end
U = U(:, 1:N-1);                   % last one has eigenvalue 0 (data was centred)

%% ===================== EIGENFACES =====================
Wpca_eig = U(:, 1:k);              % n x k: the eigenfaces

trainFeat = Wpca_eig' * A;                         % k x N
testFeat  = Wpca_eig' * (testImages - meanFace);   % k x (number of tests)

% Nearest neighbour: compare each test image with every training image
correct = 0;
for t = 1:size(testFeat, 2)
    bestDist = inf;
    for j = 1:N
        d = sum((testFeat(:, t) - trainFeat(:, j)).^2);
        if d < bestDist
            bestDist = d;
            bestLabel = trainLabels(j);
        end
    end
    if bestLabel == testLabels(t)
        correct = correct + 1;
    end
end
accEigen = 100 * correct / size(testFeat, 2);
fprintf('Eigenfaces  (k = %d):   accuracy = %.2f%%\n', k, accEigen);

%% ===================== FISHERFACES =====================
% Step A: PCA down to N - c dimensions (makes Sw invertible)
Wpca = U(:, 1:N-c);                % n x (N-c)
Z = Wpca' * A;                     % (N-c) x N, each image as N-c numbers
d = N - c;

% Step B: between-class (Sb) and within-class (Sw) scatter in this space.
% The overall mean of Z is zero, because A was centred.
Sb = zeros(d, d);
Sw = zeros(d, d);
for p = 1:c
    Zp = Z(:, trainLabels == p);   % this person's images
    Np = size(Zp, 2);
    mp = mean(Zp, 2);              % this person's mean

    Sb = Sb + Np * (mp * mp');     % spread of person means

    for j = 1:Np
        diff = Zp(:, j) - mp;      % image minus its own person's mean
        Sw = Sw + diff * diff';    % spread within this person
    end
end

fprintf('rank(Sw) = %d (should be N - c = %d)\n', rank(Sw), d);

% Step C: solve Sb*w = lambda*Sw*w and keep the top c-1 directions
[V2, D2] = eig(Sb, Sw);
[lam, order2] = sort(diag(D2), 'descend');
Wfld = V2(:, order2(1:c-1));       % (N-c) x (c-1)

% Step D: combine both projections into one matrix
Wopt = Wpca * Wfld;                % n x (c-1): the Fisherfaces

trainFeat = Wopt' * A;
testFeat  = Wopt' * (testImages - meanFace);

correct = 0;
for t = 1:size(testFeat, 2)
    bestDist = inf;
    for j = 1:N
        dd = sum((testFeat(:, t) - trainFeat(:, j)).^2);
        if dd < bestDist
            bestDist = dd;
            bestLabel = trainLabels(j);
        end
    end
    if bestLabel == testLabels(t)
        correct = correct + 1;
    end
end
accFisher = 100 * correct / size(testFeat, 2);
fprintf('Fisherfaces (%d dims):  accuracy = %.2f%%\n', c-1, accFisher);

%% Show the mean face, first 8 eigenfaces and first 8 Fisherfaces
figure;
subplot(3, 8, 1); imagesc(reshape(meanFace, h, w)); colormap gray; axis image off;
title('mean');
for i = 1:8
    subplot(3, 8, 8 + i);
    imagesc(reshape(Wpca_eig(:, i), h, w)); colormap gray; axis image off;
    title(['eig ' num2str(i)]);

    subplot(3, 8, 16 + i);
    imagesc(reshape(Wopt(:, i), h, w)); colormap gray; axis image off;
    title(['fish ' num2str(i)]);
end
