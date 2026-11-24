%% Eigenfaces and Fisherfaces: simple implementation
% Only basic MATLAB: loops, matrix multiplication, mean, eig, norm.
%
% Notation follows the study guide:
%   d    pixels per image          N    training images
%   c    number of people          n_j  images of person j
%   m    PCA dimension kept before doing LDA
%   S_W  within-class scatter      S_B  between-class scatter
%
% Folder layout expected:
%   dataDir/s1/1.pgm, dataDir/s1/2.pgm, ...   one subfolder per person
%
% Run on BOTH databases, because they say opposite things:
%   ORL   pose and expression vary  -> Fisherfaces do NOT help
%   Yale  illumination varies       -> Fisherfaces help a lot (the paper's claim)

clc; clear; close all;

figDir = 'figures/';
if exist(figDir, 'dir') == 0
    mkdir(figDir);
end

% name, folder, number of people, training images per person, height, width
% These settings match the study guide and the report, so the numbers printed
% here are the ones quoted there.
sets = { 'ORL',  '../ORL',          32,  6, 112,  92;
         'Yale', '../CroppedYale',  38, 40, 192, 168 };

for setNo = 1:2

dataName  = sets{setNo, 1};
dataDir   = sets{setNo, 2};
numPeople = sets{setNo, 3};    % c = number of people (classes)
numTrain  = sets{setNo, 4};    % training images per person (rest are test)
h         = sets{setNo, 5};
w         = sets{setNo, 6};

% Fail loudly if the data is not there.  dir() on a missing folder returns an
% EMPTY list rather than an error, so without this the script would carry on
% with zero images and die much later for an unrelated-looking reason.
if exist(dataDir, 'dir') == 0
    error(['Cannot find %s.  Run this from the folder holding this file, ' ...
           'with ORL/ and CroppedYale/ one level above it.'], dataDir);
end

fprintf('\n================ %s ================\n', dataName);

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

d = size(trainImages, 1);          % number of pixels
N = size(trainImages, 2);          % number of training images
c = numPeople;                     % number of classes
fprintf('d = %d pixels, N = %d training, %d test, c = %d people\n', ...
        d, N, size(testImages, 2), c);

%% 2. Subtract the mean face
meanFace = mean(trainImages, 2);   % d x 1
A = trainImages - meanFace;        % d x N, centred training images

%% 3. PCA using the small matrix L = A'*A (N x N instead of d x d)
% For Yale the covariance would be 32256 x 32256 = 8.3 GB.  We never form it.
L = A' * A;                        % N x N
[V, D] = eig(L);                   % eigenvectors of L
[vals, order] = sort(diag(D), 'descend');   % eig is ASCENDING; flip it
V = V(:, order);

% If L*v = lambda*v, then (A*A')*(A*v) = lambda*(A*v),
% so A*v is an eigenvector of the big covariance matrix with the same lambda.
U = A * V;                         % d x N
for i = 1:N
    U(:, i) = U(:, i) / norm(U(:, i));   % ||A*v|| = sqrt(lambda), not 1
end
U = U(:, 1:N-1);                   % last one has eigenvalue 0 (data was centred)

%% 4. CORRELATION: the baseline with no projection at all
% Each image is made zero mean and unit norm first.  That turns nearest
% neighbour into CORRELATION matching, because for unit vectors
%       ||a-b||^2 = 2 - 2*a'*b
% so the nearest neighbour is the most correlated one.  If a projection
% cannot beat this, it is not earning its place.
trN = normaliseColumns(trainImages);
teN = normaliseColumns(testImages);
accCorr = nearestNeighbour(trN, trainLabels, teN, testLabels);
fprintf('Correlation (raw pixels):  accuracy = %.2f%%\n', accCorr);

%% 5. EIGENFACES
% Try several k and keep the best, then also report k = c-1 so the
% comparison with Fisherfaces uses the SAME number of features.
ks = [10 20 30 50 80 120 170];
ks = ks(ks <= N - 1);
accBest = 0; bestK = 0;
for kk = ks
    Uk = U(:, 1:kk);
    r = nearestNeighbour(Uk'*A, trainLabels, Uk'*(testImages-meanFace), testLabels);
    if r > accBest
        accBest = r;
        bestK   = kk;
    end
end
fprintf('Eigenfaces  (best k = %d): accuracy = %.2f%%\n', bestK, accBest);

Uc = U(:, 1:(c-1));
accEigMatched = nearestNeighbour(Uc'*A, trainLabels, Uc'*(testImages-meanFace), testLabels);
fprintf('Eigenfaces  (k = c-1 = %d): accuracy = %.2f%%\n', c-1, accEigMatched);

% The paper's other baseline: throw away the first three eigenfaces.  Under
% Lambertian reflectance the images of one face under all lightings span a
% 3-D subspace, so components 1-3 are expected to be mostly illumination.
% Dropping them should HELP where lighting is the nuisance (Yale) and HURT
% where the leading components carry identity (ORL).
accDrop3 = 0; bestK3 = 0;
for kk = ks
    if kk + 3 > size(U, 2), continue; end
    Uk3 = U(:, 4:(kk+3));          % skip the first three
    r = nearestNeighbour(Uk3'*A, trainLabels, Uk3'*(testImages-meanFace), testLabels);
    if r > accDrop3
        accDrop3 = r;
        bestK3   = kk;
    end
end
fprintf('Eigenfaces  w/o top 3 (k = %d): accuracy = %.2f%%\n', bestK3, accDrop3);

%% 6. FISHERFACES
% Step A: PCA down to m = N - c dimensions, which makes S_W invertible.
% In pixel space rank(S_W) <= N - c, far below d, so S_W is singular there
% and J(w) is not even defined.  Projecting first throws away exactly the
% directions S_W knows nothing about.
m = N - c;
Wpca = U(:, 1:m);                  % d x m
Z = Wpca' * A;                     % m x N, each image as m numbers

% Step B: the two scatter matrices in this space.
% The overall mean of Z is zero, because A was already centred, which is why
% S_B below uses mp directly rather than (mp - overall mean).
Sb = zeros(m, m);
Sw = zeros(m, m);
for p = 1:c
    Zp = Z(:, trainLabels == p);   % this person's images
    Np = size(Zp, 2);
    mp = mean(Zp, 2);              % this person's mean

    Sb = Sb + Np * (mp * mp');     % spread of the person means

    for j = 1:Np
        diff = Zp(:, j) - mp;      % image minus its own person's mean
        Sw = Sw + diff * diff';    % spread within this person
    end
end

fprintf('rank(S_B) = %d (bound c-1 = %d), rank(S_W) = %d (bound N-c = %d)\n', ...
        rank(Sb), c-1, rank(Sw), m);

% Step C: solve S_B*w = lambda*S_W*w and keep the top c-1 directions.
% rank(S_B) <= c-1, so everything past c-1 has eigenvalue zero and its
% eigenvector is an arbitrary basis of a null space -- pure noise.
% (The study guide derives the Cholesky whitening route by hand; we measured
%  the two agreeing to 0.00 degrees, so the one-liner is used here.)
[V2, D2] = eig(Sb, Sw);
[lam, order2] = sort(diag(D2), 'descend');
Wfld = V2(:, order2(1:c-1));       % m x (c-1)

% Step D: combine both projections into one matrix
Wopt = Wpca * Wfld;                % d x (c-1): the Fisherfaces

% Step E: make each Fisherface unit length.  THIS IS NOT COSMETIC.
% eig(Sb,Sw) returns columns with W'*S_W*W = I, NOT unit Euclidean length,
% so their lengths differ by a factor of ~287 here.  J(w) does not care, but
% nearest neighbour with Euclidean distance does: the long directions would
% dominate every comparison.  Skipping these lines costs ORL 88.5% -> 29.5%.
for i = 1:size(Wopt, 2)
    Wopt(:, i) = Wopt(:, i) / norm(Wopt(:, i));
end

accFisher = nearestNeighbour(Wopt'*A, trainLabels, ...
                             Wopt'*(testImages-meanFace), testLabels);
fprintf('FISHERFACES (%d dims):      accuracy = %.2f%%\n', c-1, accFisher);

%% 7. Show the mean face, first 8 eigenfaces and first 8 Fisherfaces
figure('Position', [100 100 900 400]);
subplot(3, 8, 1); imagesc(reshape(meanFace, h, w)); colormap gray; axis image off;
title('mean');
for i = 1:8
    subplot(3, 8, 8 + i);
    imagesc(reshape(U(:, i), h, w)); colormap gray; axis image off;
    title(['eig ' num2str(i)]);

    subplot(3, 8, 16 + i);
    imagesc(reshape(Wopt(:, i), h, w)); colormap gray; axis image off;
    title(['fish ' num2str(i)]);
end
sgtitle([dataName ': eigenfaces are smooth, Fisherfaces are high-frequency']);
saveas(gcf, [figDir dataName '_faces.png']);

end

fprintf('\nfigures saved to %s\n', figDir);


%% ---------------- helper functions ----------------

function rate = nearestNeighbour(trainFeat, trainLabels, testFeat, testLabels)
% Compare each test column with every training column and take the label of
% the closest.  Squared distances: sqrt is increasing, so it cannot change
% which neighbour is nearest.
correct = 0;
for t = 1:size(testFeat, 2)
    bestDist = inf;
    for j = 1:size(trainFeat, 2)
        dist = sum((testFeat(:, t) - trainFeat(:, j)).^2);
        if dist < bestDist
            bestDist  = dist;
            bestLabel = trainLabels(j);
        end
    end
    if bestLabel == testLabels(t)
        correct = correct + 1;
    end
end
rate = 100 * correct / size(testFeat, 2);
end


function Xn = normaliseColumns(X)
% Make every image zero mean and unit norm.
Xn = X - mean(X, 1);
for i = 1:size(Xn, 2)
    nrm = norm(Xn(:, i));
    if nrm > 0
        Xn(:, i) = Xn(:, i) / nrm;
    end
end
end
