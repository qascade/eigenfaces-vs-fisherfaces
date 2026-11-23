function Xn = saumya_normcols(X)
% SAUMYA_NORMCOLS  Make every image (column) zero-mean and unit-norm.
%
%   This is what turns nearest neighbour into a CORRELATION classifier: for
%   zero-mean unit-norm a and b,
%         ||a-b||^2 = 2 - 2*a'*b
%   so the nearest neighbour is exactly the most correlated one.  It also
%   removes each image's brightness offset and contrast scaling.

Xn = X - mean(X, 1);           % mean(X,1) is the per-COLUMN (per-image) mean
for i = 1:size(Xn, 2)
    nrm = norm(Xn(:, i));
    if nrm > 0
        Xn(:, i) = Xn(:, i) / nrm;
    end
end
end
