function [Wopt, Sb, Sw] = saumya_fisherfaces(V, xbar, imgs, y, m, c)
% SAUMYA_FISHERFACES  The two-stage algorithm: PCA to m dimensions, then LDA.
%
%   V, xbar   the PCA basis and mean face from saumya_pca
%   imgs, y   the training images and their labels
%   m         how many PCA directions to keep before doing LDA
%   c         number of classes
%
%   Wopt      d x (c-1), the Fisherfaces as columns, unit length
%   Sb, Sw    the scatter matrices as built in the reduced space
%
%   WHY TWO STAGES.  In pixel space S_W is d x d but rank(S_W) <= N-c.  For
%   Yale that is 1482 << 32256, so S_W is singular, S_W^-1 does not exist,
%   and J(w) = (w'*S_B*w)/(w'*S_W*w) is undefined for any w in the null
%   space.  Projecting onto m <= N-c PCA directions throws away exactly the
%   directions S_W knows nothing about.  What is left has full rank m.

% m is squeezed from BOTH sides:  c-1 <= m <= N-c.
% Above, so that S_W is invertible.  Below, because we pull c-1 directions
% out of an m-dimensional space and cannot have more directions than
% dimensions.
if m < c - 1
    error(['saumya_fisherfaces: m = %d is below c-1 = %d. ' ...
           'Cannot extract %d Fisherfaces from a %d-dimensional space.'], ...
          m, c-1, c-1, m);
end

Vm   = V(:, 1:m);              % the PCA basis we work in
Al   = Vm' * (imgs - xbar);    % the training data in m dimensions
abar = mean(Al, 2);

% ---- the two scatter matrices, straight from the definitions ----
Sw = zeros(m, m);
Sb = zeros(m, m);
for j = 1:c
    Aj = Al(:, y == j);        % logical indexing: all columns of person j
    nj = size(Aj, 2);
    mj = mean(Aj, 2);          % this person's mean
    Dj = Aj - mj;              % centred on THEIR OWN mean
    Sw = Sw + Dj * Dj';
    Sb = Sb + nj * (mj - abar) * (mj - abar)';
end

% Floating point makes these very slightly non-symmetric (about 1e-16).
% chol and symmetric eig read only ONE triangle, so force exact symmetry
% rather than letting the solver silently pick a half.
Sw = (Sw + Sw') / 2;
Sb = (Sb + Sb') / 2;

% ---- solve  S_B*w = lambda*S_W*w  by Cholesky whitening ----
%   S_W = R'*R with R upper triangular.  Substituting w = inv(R)*z:
%        S_B*inv(R)*z = lambda*R'*R*inv(R)*z = lambda*R'*z
%   then multiplying on the left by inv(R'):
%        inv(R')*S_B*inv(R)*z = lambda*z        <- call this M
%   M is symmetric because M' = inv(R')*S_B'*inv(R) = M, so a GENERALISED
%   eigenproblem has become an ORDINARY SYMMETRIC one: eigenvalues are
%   guaranteed real and eigenvectors orthogonal.
R = chol(Sw);

% (R')\(Sb/R) is inv(R')*Sb*inv(R) done as two TRIANGULAR SOLVES, without
% ever forming an inverse.  Sb/R solves from the right, (R')\... from the left.
M = (R') \ (Sb / R);
M = (M + M') / 2;

[Z, Dz] = eig(M);
[~, idx] = sort(diag(Dz), 'descend');      % eig is ascending; best first
Z = Z(:, idx);

% Keep c-1 directions only: rank(S_B) <= c-1, so later eigenvalues are zero
% and their eigenvectors are an arbitrary basis of a null space.
Wfld = R \ Z(:, 1:(c-1));      % map back: w = inv(R)*z, i.e. solve R*w = z
Wopt = Vm * Wfld;              % compose the two stages

% ---- unit-norm each Fisherface.  THIS IS NOT COSMETIC ----
% J(w) is scale invariant so the SUBSPACE is already right.  But nearest
% neighbour with Euclidean distance is NOT scale invariant, and these columns
% satisfy W'*S_W*W = I, which makes their lengths differ by orders of
% magnitude (whitening scales direction l by 1/sqrt(lambda_l)).  Skipping
% this costs ORL 90.6% -> 20.3% and Yale 67.2% -> 13.5%.
for l = 1:size(Wopt, 2)
    nrm = norm(Wopt(:, l));
    if nrm > 0
        Wopt(:, l) = Wopt(:, l) / nrm;
    end
end
end
