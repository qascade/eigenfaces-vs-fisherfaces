function [imTr, ytr, imTe, yte, teNames, trNames] = saumya_loadfaces(folders, numTrain, d)
% SAUMYA_LOADFACES  Read PGM images, one image per column, split train/test.
%
%   folders   cell array of folder paths, one per person (= one per class)
%   numTrain  how many images of each person go into training
%   d         pixels per image
%
%   imTr  d x N   training images, ONE IMAGE PER COLUMN
%   ytr   1 x N   ytr(i) is which person is in column i
%   imTe, yte     the same for the test images
%   teNames       filename of each test column (needed to read lighting angle)
%   trNames       filename of each training column
%
%   The class label is the LOOP INDEX p, never the folder name.  That is why
%   the folders can be called anything.
%
%   GOTCHA: dir() sorts names as TEXT.  For ORL that means 1,10,2,3,... so
%   "the first 6" is really {1,10,2,3,4,5}.  For Yale the names encode
%   lighting as A+020 / A-020 and '+' sorts before '-', so training takes the
%   positive azimuths and testing gets the negative ones.

nF = numel(folders);

% Count first so we can preallocate.  Growing an array inside a loop makes
% MATLAB copy the whole thing every iteration.
nTr = 0;
nTe = 0;
for p = 1:nF
    f = dir([folders{p} '*.pgm']);
    if isempty(f)
        error('saumya_loadfaces: no .pgm images in %s', folders{p});
    end
    ntr = min(numTrain, numel(f));
    nTr = nTr + ntr;
    nTe = nTe + (numel(f) - ntr);
end

imTr = zeros(d, nTr);   ytr = zeros(1, nTr);   trNames = cell(1, nTr);
imTe = zeros(d, nTe);   yte = zeros(1, nTe);   teNames = cell(1, nTe);

a = 0;
b = 0;
for p = 1:nF
    files = dir([folders{p} '*.pgm']);
    for j = 1:numel(files)
        im = double(imread([folders{p} files(j).name]));
        if j <= numTrain
            a = a + 1;
            imTr(:, a)  = im(:);        % im(:) stacks the image COLUMN by COLUMN
            ytr(a)      = p;
            trNames{a}  = files(j).name;
        else
            b = b + 1;
            imTe(:, b)  = im(:);
            yte(b)      = p;
            teNames{b}  = files(j).name;
        end
    end
end
end
