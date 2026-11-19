# `fisherfaces_project.m` — line-by-line walkthrough

Companion to `study_guide.pdf`. The PDF carries the theory; this file explains the
code line by line: **what it does**, **what breaks without it**, and **what else we
could have written**.

Read the PDF's Part IV first if you are unsure about MATLAB syntax itself — this
file assumes you know what `im(:)` and `dir` do and focuses on *why each line is
there*.

---

## S0 — Configuration

```matlab
clc; clear; close all;
```
`clc` clears the console, `clear` removes all variables, `close all` closes
figures. **Why it matters:** without `clear`, a stale `trainImages` from a previous
run can silently survive and be reused, so the script appears to work while
reporting numbers from old data. This is the single most common way to fool
yourself in MATLAB.

```matlab
orlRoot  = '../ORL/';
yaleRoot = '../CroppedYale/';
```
Relative to `hw3/fisherfaces_project/`. *Alternative:* absolute paths — more
robust but not portable to the other group member's machine.

```matlab
if exist(figDir, 'dir') == 0
    mkdir(figDir);
end
```
`exist(...,'dir')` returns 7 for a directory, 0 if absent. **Without this**,
`saveas` fails at the very end of a six-minute run — the worst possible place.

```matlab
rng(0);
```
Seeds the random number generator. **Essential here:** two experiments use random
splits (`'random'` mode and the n-sweep). Without a fixed seed every run gives
different numbers and nothing in the report is reproducible. *Alternative:*
`rng('shuffle')` plus averaging over many repeats — statistically better, but then
we cannot quote one exact number per cell.

---

## S1 — Toy 2-D experiment

```matlab
Sigma = [16 0; 0 0.25];
mu1   = [0; -1];
mu2   = [0;  1];
```
Deliberately constructed so PCA and LDA disagree: within-class variance is 16 along
x and 0.25 along y, but the classes are separated **only** along y. Total variance
along y is `0.25 + 1 = 1.25`, still far below 16, so PCA picks x.

```matlab
R = chol(Sigma);
X1(:, i) = mu1 + R' * randn(2, 1);
```
To sample from `N(mu, Sigma)` we need a matrix `Rt` with `Rt*Rt' = Sigma`. `chol`
gives `Sigma = R'*R`, so `Rt = R'`. Check: if `z ~ N(0,I)` then
`cov(R'z) = R'*I*R = R'*R = Sigma`. ✓

**Gotcha:** using `R` instead of `R'` gives `cov(Rz) = R*R'`, which is *not*
`Sigma` unless `Sigma` is diagonal. Ours *is* diagonal, so this particular bug
would be invisible here — which is exactly why it is worth getting right on
purpose.

```matlab
SwToy = D1 * D1' + D2 * D2';
```
Within-class **scatter** — a sum of outer products with no `1/n`. Since the Fisher
criterion is a *ratio*, any constant factor cancels, so scatter and covariance give
the same direction. We use scatter to match the theory in the PDF.

```matlab
wLDA = SwToy \ (m2 - m1);
```
The closed form `w ∝ inv(Sw)(m2-m1)`. The backslash **solves** the system instead
of forming the inverse — faster and numerically better. *Alternative:*
`inv(SwToy)*(m2-m1)` works here (2×2, well-conditioned) but is the habit that
causes real damage at m=1482.

```matlab
for i = 1:1000
    wr = randn(2, 1);  wr = wr / norm(wr);
```
A sanity floor: the best of 1000 random directions. We measured 4.187249 against
LDA's 4.187377 — LDA wins, but only just, because in 2-D a random direction
sometimes lands near optimal. **This is why the brute-force angle sweep below is the
stronger check.**

```matlab
for t = 0:0.0005:pi
    wt = [cos(t); sin(t)];
```
Every unit direction in 2-D is `(cos t, sin t)` for `t ∈ [0, π)` (beyond π we just
get the same lines negated). 6283 angles. This is a **direct numerical proof** that
the closed form maximises J, and it needs no toolbox. We added it precisely because
`fitcdiscr` is unavailable on this machine.

```matlab
bayesErr = 0.5 * erfc((delta / 2) / sqrt(2));
```
`normcdf` needs the Statistics Toolbox, which we do not have. Uses
`Φ(-a) = ½·erfc(a/√2)`. **Do not** write `erf` instead of `erfc` — they differ by
`erfc(x) = 1 - erf(x)`, and the result would be `0.9772` instead of `0.0228`.

---

## S2 — Folder lists

```matlab
for p = 1:40
    orlFolders40{p} = [orlRoot 's' num2str(p) '/'];
end
```
Paths built **by index**, not by globbing. **Why:** `ORL/` contains a stray empty
directory called `sed`, so `dir('../ORL/s*')` returns 41 entries and a glob-based
loader would try to read images out of it and crash (or, worse, produce an empty
class). Index construction never sees it.

```matlab
dy = dir(yaleRoot);
if dy(i).isdir && dy(i).name(1) ~= '.'
```
Yale **must** be listed rather than index-built, because `yaleB14` does not exist —
the names are not a contiguous range. The filter drops `.` and `..`, and the
`name(1) ~= '.'` form also drops macOS hidden files like `.DS_Store` as a bonus.

*Alternative:* `regexp(name, '^yaleB\d+$')` — more precise but we avoided `regexp`
deliberately, since we must be able to explain every construct.

---

## S3/S4 — Eigenfaces baseline (regression test)

The purpose of this section is **not** to produce new science. It re-derives HW3's
numbers to prove the rewritten loader did not change the data. Expected:

```
ORL-32 : 17.2 44.5 71.1 75.8 93.0 92.2 92.2 94.5 95.3 94.5 93.8 93.8 93.0
```

If these drift, the split logic diverged and nothing downstream is trustworthy.
**Run this section first whenever anything in the loader changes.**

```matlab
rateO(t) = nnClassify(Uk' * AO, ytrO, Uk' * AteO, yteO);
```
Note `AteO = XteO - meanO` uses the **training** mean, not the test mean. This is
important and easy to get wrong: at test time you do not have access to the test
set's statistics. Using the test mean would be information leakage.

---

## S5 — Why `S_W` cannot be inverted

```matlab
AwY(:, cols) = XtrY(:, cols) - mi;
```
Builds the within-class-centred matrix: each image minus **its own class mean**
(not the global mean). This is the matrix whose Gram we take.

```matlab
GwY = AwY' * AwY;
rankSwY = rank(GwY);
```
Uses `rank(Aw*Aw') = rank(Aw) = rank(Aw'*Aw)` to get the rank of an
`8.3 GB` matrix from a `1520×1520` one. **Never** write `Aw*Aw'` here — it would
attempt to allocate 8.3 GB and die.

```matlab
if norm(XtrY(:, cols(u)) - XtrY(:, cols(v))) == 0
```
Duplicate search. We added this after measuring `rank = 1481` instead of the
predicted 1482. It found one byte-identical pair in subject 12, caused by a stray
`yaleB12_P00A+020E+10 - Copy.pgm` in the dataset. Two identical columns stay
identical after centring, hence an exact linear dependency costing one rank.
**That file has since been removed and the rank now meets the bound exactly.**

`== 0` is an exact float comparison, which is normally bad practice — here it is
correct *because* we are looking for byte-identical duplicates, not near-duplicates.

```matlab
clear AwY GwY;
```
`AwY` is 392 MB. Without this the memory stays occupied for the rest of the run.

---

## S6/S7/S8 — Main comparison

```matlab
configs = { ...
    'ORL-32  (HW3 split)',  {orlFolders32}, 6,  'hw3',    dORL,  orlH, orlW; ...
```
A cell array acting as a table of experiments. The nested `{orlFolders32}` is
needed because the outer cell would otherwise *flatten* the inner cell array of
folder paths; wrapping it keeps it as a single element. (`configs{ci,2}{1}` unwraps
it.) **This is a real MATLAB trap** and the error it produces is confusing.

```matlab
rCorr = nnClassify(normCols(Xtr), ytr, normCols(Xte), yte);
```
The correlation baseline from the paper. `normCols` makes each image zero-mean and
unit-norm, which turns Euclidean nearest-neighbour into maximum-correlation
matching (`‖a-b‖² = 2 - 2a'b`).

```matlab
if k > 3
    rEig3(t) = nnClassify(P(4:k, :), ytr, Q(4:k, :), yte);
end
```
The "drop the top 3" variant: keep **rows** 4..k of the already-projected
coefficients. Note this is cheaper and cleaner than rebuilding `U(:,4:k)` and
re-projecting, and gives the identical answer because projection is applied
per-row. The `if k > 3` guard exists because `P(4:3,:)` would be empty and
`nnClassify` would divide by zero.

```matlab
Ucm1 = U(:, 1:(c-1));
rEigMatched = nnClassify(Ucm1' * A, ytr, Ucm1' * Ate, yte);
```
**The fairness line.** Fisherfaces gets exactly `c-1` features, so we also measure
Eigenfaces at `c-1`. Quoting Eigenfaces' best-over-all-k against Fisherfaces' fixed
`c-1` would be comparing 170 features to 37.

```matlab
m = N - c;
[Wopt, info] = fisherFromPCA(U, mf, Xtr, ytr, m, 2);
```
`m = N-c` is the paper's choice. The `2` requests full diagnostics, including the
cross-check against MATLAB's `eig(Sb,Sw,'chol')`. In the sweeps we pass `1` or `0`
because that cross-check doubles the cost.

```matlab
clear U A Ate Xtr Xte Wopt;
```
Freed at the end of every configuration. Without it, five Yale-sized
configurations would accumulate several GB and the script would thrash or die.

---

## S9 — Sweep of the PCA dimension `m`

The key experiment. `fisherFromPCA` takes a **precomputed** `U` precisely so this
sweep does not recompute the expensive eigendecomposition for every `m` — only the
scatter matrices and the small eigenproblem are redone.

```matlab
yyaxis left;  plot(S.mList, S.rate, ...)
yyaxis right; semilogy(S.mList, S.cond, ...)
```
Two y-scales: recognition rate (linear) against `cond(S_W)` (log). The log scale is
mandatory — the condition number spans about nine orders of magnitude.

---

## S10 — Sweep of training images per subject

```matlab
[Xtr, ytr, Xte, yte] = loadFaces(yaleFolders, nTr, 'random', dYALE);
```
Uses the **random** split, not the HW3 lexicographic one. **Why:** the
lexicographic split has a severe lighting bias (see the PDF, §13.2). If we used it
here, the training-size effect and the split bias would be tangled together and the
curve would be uninterpretable.

```matlab
ksHere = ksHere(ksHere <= size(U, 2));
```
At `n = 2` we have only `N = 76` images, hence at most 76 eigenfaces. Asking for
`k = 170` would index out of bounds. This line clips the list to what exists.

---

## S11 — Figures

```matlab
figure('Visible', 'off');
```
Required under `matlab -batch`: there is no display, and a visible figure would
either error or hang.

```matlab
imagesc(reshape(keepORL.U(:, i), orlH, orlW));
colormap gray; axis image off;
```
- `reshape(..., 112, 92)` — the inverse of `im(:)`. Height first, then width. Swap
  them and you get a scrambled image with **no error**.
- `imagesc` auto-scales to the data range. Eigenfaces contain small positive and
  negative values, so `image` (no scaling) would render them nearly black.
- `axis image` forces square pixels so faces are not stretched; `off` hides ticks.

---

## S12 — Numerical checks

```matlab
norm(U20' * U20 - eye(20), 'fro')
```
(a) Orthonormality. Measured `9.06e-14`. If the normalisation loop in `doPCA` were
removed, this would be enormous.

```matlab
agree(i) = abs(Usvd(:, i)' * keepORL.U(:, i));
```
(b) `eig(A'A)` vs `svd(A)`. **`abs` is essential** — eigenvector signs are
arbitrary, so a correct implementation can return `-u`. Comparing `norm(u1-u2)`
would report a spurious failure half the time.

```matlab
G_eu = Wf' * Wf;
G_sw = WfY' * SwO * WfY;
```
(c) The orthogonality check. Measured: half the energy of `W'W` is off-diagonal,
while `W'SwW` is diagonal to `1e-13`. Fisherfaces are **S_W-orthogonal, not
Euclidean-orthogonal**.

```matlab
rank(SbO)
```
(d) Confirms `rank(S_B) = c-1 = 31` exactly, matching the theorem.

---

## Local functions

### `loadFaces`

**Two passes.** Pass 1 counts files so we can preallocate; pass 2 reads them.
HW3 used `X = [X, im(:)]` inside the loop, which reallocates and copies the entire
matrix on **every image** — 1520 copies of a 392 MB array for Yale.

```matlab
ntr = min(numTrain, nf);
```
Guards against a subject with fewer images than `numTrain`. Yale subjects have
59–64 images, not a constant 64, so **nothing may hardcode 64**.

```matlab
vals(j) = sscanf(files(j).name, '%d');
```
Numeric sort for ORL. Safe only because ORL filenames *start* with a digit;
`sscanf('yaleB01_...', '%d')` returns empty, which is why `'numeric'` mode is never
used for Yale.

```matlab
Xtr(:, a) = im(:);
```
One image per **column**. The whole script depends on this orientation; the
d×N-vs-N×d choice propagates into every matrix product and every `mean(X,2)`.

### `doPCA`

```matlab
L = A' * A;
[V, D] = eig(L);
[eigvals, idx] = sort(diag(D), 'descend');
```
The Gram trick plus the mandatory descending sort. `eig` returns **ascending**
order, so without the sort we would keep the *least* informative directions —
silently, with no error, at chance-level accuracy.

```matlab
U = A * V;
for i = 1:size(U, 2)
    nrm = norm(U(:, i));
    if nrm > 1e-10
        U(:, i) = U(:, i) / nrm;
    end
end
```
`A*v` has norm `sqrt(λ)`, not 1. The `if` guard matters: after mean subtraction
`rank(A) ≤ N-1`, so the last eigenvalue is ~0 and its column is pure round-off.
Dividing that by its own tiny norm would amplify noise into a full-amplitude
garbage image.

```matlab
lambda = eigvals / M;
```
Eigenvalues of `L = A'A` are `M×` those of `C = AA'/M`. **Without this line the
recognition rates are unchanged but the reconstruction-error identity is wrong by a
factor of 192** — which is exactly what makes it a dangerous bug.

### `nnClassify`

```matlab
distances = sum((trainProj - testProj(:, i)).^2, 1);
[~, best] = min(distances);
```
Implicit expansion subtracts the test vector from every training column at once;
`sum(...,1)` collapses each column to one squared distance. No `sqrt` — monotonic,
so it cannot change the argmin. `.^` is element-wise; plain `^` would be a matrix
power and wrong.

*Alternative:* `pdist2` (Statistics Toolbox — not installed here) or a fully
vectorised `‖a‖² - 2a'b + ‖b‖²` expansion. The loop is O(N) per test image either
way and is far easier to defend line by line.

### `normCols`

```matlab
Xn = X - mean(X, 1);
```
`mean(X,1)` is the **per-image** mean (one number per column), not the mean face.
Removing it plus unit-normalising makes nearest-neighbour equivalent to maximum
correlation.

### `fisherFromPCA`

```matlab
Um = U(:, 1:m);
Y  = Um' * A;
```
Stage 1: project to `m` dimensions. Takes `U` as an **argument** so the m-sweep
does not recompute PCA each time.

```matlab
Sw = Sw + Di * Di';
Sb = Sb + ni * (mi - muY) * (mi - muY)';
```
The definitions, accumulated class by class. Note `Sb` is weighted by `ni` —
dropping that weight gives *unweighted* LDA, a different (and for unequal class
sizes, wrong) method.

```matlab
Sw = (Sw + Sw') / 2;
Sb = (Sb + Sb') / 2;
```
Floating-point addition is not associative, so the accumulated matrices differ from
their transposes by ~1e-16. `chol` and symmetric `eig` read **one triangle** and
assume the other matches.

```matlab
info.resST = norm(St - (Sb + Sw), 'fro') / norm(St, 'fro');
```
The free correctness test `S_T = S_B + S_W`. Measured ~1e-16 everywhere. A wrong
class mean or count would break it immediately.

```matlab
[R, p] = chol(Sw);
if p ~= 0
    ridge = 1e-6 * trace(Sw) / m;
```
**Two-output `chol`** so failure is a return value, not an exception that aborts
the sweep. The ridge is scaled by `trace(Sw)/m` — the *mean eigenvalue* — so it is
comparable to `S_W`'s own magnitude. A bare `1e-6*eye(m)` would be overwhelming on
one dataset and negligible on another.

```matlab
Minner = (R') \ (Sb / R);
```
`inv(R')*Sb*inv(R)` without ever forming an inverse. `Sb / R` solves from the
right; `(R') \ X` solves from the left. The result is symmetric (see the PDF
derivation), which is the entire point of the manoeuvre.

```matlab
Wfld = R \ Z(:, 1:(c-1));
```
Maps back: `w = inv(R)*z`. Truncated to `c-1` because `rank(S_B) ≤ c-1`, so the
remaining eigenvalues are zero and their eigenvectors are arbitrary noise.

```matlab
Wopt = Um * Wfld;
```
Composes both stages into one `d × (c-1)` projection.

```matlab
Wopt(:, i) = Wopt(:, i) / nrm;
```
A **convention**, not mathematics. `J` is scale-invariant so LDA does not fix the
scaling, but nearest-neighbour distances *do* depend on it. Unit norm is the usual
choice. Different conventions give slightly different recognition rates — worth
knowing before being asked.
