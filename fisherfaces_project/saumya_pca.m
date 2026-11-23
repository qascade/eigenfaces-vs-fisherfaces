function [V, lambda, xbar] = saumya_pca(imgs)
% SAUMYA_PCA  Eigenfaces, using the L = X'*X trick.
%
%   imgs    d x N, one image per column
%   V       d x N, the eigenfaces as COLUMNS, best first, unit length
%   lambda  the eigenvalues of the covariance, descending
%   xbar    the mean face
%
%   WHY THE TRICK.  The covariance is C = X*X'/N, which is d x d.  For Yale
%   that is 32256 x 32256 = 8.3 GB, impossible to store.  But if L = X'*X
%   (only N x N) satisfies L*w = lambda*w, then multiplying by X gives
%        X*X'*(X*w) = X*(X'*X*w) = lambda*(X*w)
%   so X*w is an eigenvector of X*X' with the SAME eigenvalue.  Solve the
%   small problem, multiply by X, done.
%
%   CATCH: ||X*w|| = sqrt(lambda), not 1, so the columns need renormalising.

[~, N] = size(imgs);

xbar = mean(imgs, 2);          % the mean face
X    = imgs - xbar;            % centre every image

L = X' * X;                    % N x N instead of d x d
[Wg, D] = eig(L);

% eig returns eigenvalues ASCENDING.  We want the largest first.
[eigvals, idx] = sort(diag(D), 'descend');
Wg = Wg(:, idx);

V = X * Wg;

% Renormalise.  After centring, rank(X) <= N-1, so the last direction is
% numerically empty; dividing by its tiny norm would turn round-off into a
% unit vector.  The threshold is RELATIVE to the first column.
scale = norm(V(:, 1));
for l = 1:size(V, 2)
    nrm = norm(V(:, l));
    if nrm > 1e-10 * scale
        V(:, l) = V(:, l) / nrm;
    else
        V(:, l) = 0;
    end
end

lambda = eigvals / N;          % eigenvalues of L are N times those of C
end
