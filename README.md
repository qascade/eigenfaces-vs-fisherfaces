# Eigenfaces vs. Fisherfaces

A from-scratch MATLAB implementation and study of

> P. N. Belhumeur, J. P. Hespanha, D. J. Kriegman,
> *"Eigenfaces vs. Fisherfaces: Recognition Using Class Specific Linear Projection"*,
> IEEE TPAMI 19(7), 1997.

CS 740 course project — Shubh Karman Singh (26M0843), Saumya Bajpai (26D0366).

Everything is implemented directly from the linear algebra: PCA via the
`L = AᵀA` trick, Fisher's LDA via Cholesky whitening of the generalised
eigenproblem, and nearest-neighbour classification. No machine-learning toolbox
is used — `fitcdiscr` and `pca` appear only as optional cross-checks.

---

## Results

Recognition rate (%), 1-nearest-neighbour. Fisherfaces always uses exactly
`c−1` features, so the matched-dimension Eigenfaces column is the fair comparison.

| Dataset | Correlation | Eigenfaces (best k) | Eigenfaces @ k=c−1 | Eigenfaces w/o top 3 | **Fisherfaces** |
|---|---|---|---|---|---|
| ORL-32 (HW3 split) | 91.4 | 95.3 | 94.5 | 87.5 | 90.6 |
| ORL-40 (HW3 split) | 91.2 | 93.8 | 94.4 | 86.9 | 84.4 |
| ORL-32 (numeric split) | 93.0 | 95.3 | 95.3 | 89.1 | 93.0 |
| Yale-38 (HW3 split) | 43.0 | 34.2 | 29.5 | 58.7 | **67.2** |
| Yale-38 (random split) | 85.2 | 70.0 | 50.2 | 89.0 | **97.1** |

**The paper's claim reproduces on Yale**, where the nuisance is illumination:
Fisherfaces nearly doubles Eigenfaces (67.2 vs 34.2), and on a random split cuts
the error rate from 30.0% to 2.9%.

**It fails on ORL**, where variation is pose and expression and there are only 6
images per subject. This is reported rather than tuned away: `rank(S_W) ≤ N−c`
means each class contributes only `nᵢ−1 = 5` directions to the within-class
scatter, so `S_W` is badly estimated and LDA then inverts it. Yale has 39
directions per class and wins comfortably.

### Three findings beyond the paper

1. **`m = N−c` is the weakest part of the recipe.** It is the largest PCA
   dimension that makes `S_W` invertible, but also the *worst-conditioned* such
   choice — at `m = N−c` the rank equals the dimension exactly, with no slack, so
   the smallest eigenvalues are estimation noise and `S_W⁻¹` amplifies them.
   Sweeping `m` beats the paper's choice on both databases (ORL 95.3 vs 90.6 at
   m=60; Yale 97.0 vs 95.7 at m=1000), and `cond(S_W)` rises from 2.6e6 to 2.2e11
   over the last 282 dimensions. On one split it does not even deliver the
   invertibility it was chosen for.

2. **The lexicographic Yale split is badly biased.** `dir` sorts alphabetically
   and `'+' < '-'`, so taking "the first 40 files" trains on all 35
   positive-azimuth lightings and tests on the 24 most extreme negative ones.
   Randomising the split roughly doubles *every* method. The method *ranking* is
   unchanged, so the paper's claim is unaffected.

3. **A duplicated image was detectable from the rank alone.** `rank(S_W)`
   measured 1481 against the theoretical bound `N−c = 1482`. The cause was a stray
   byte-identical file in `CroppedYale/yaleB12/`; two identical columns stay
   identical after within-class centring, costing exactly one unit of rank.

---

## Repository layout

```
final_submission/              <- the submission: code, report, figures
  toy.m                        the 2-D toy problem: why PCA is not a classifier
  eigenfaces_vs_fisherfaces.m  the main comparison, on ORL and Yale
  experiments.m                experiments E1 to E7
  figures/                     all eleven figures the scripts draw
  report.pdf / report.tex      the report
  study_guide.pdf              full derivations and a question bank
  run_log_*.txt                the output every quoted number comes from
  orl_glasses_labels.txt       glasses labels for ORL, made by inspection
  rename_yale.sh               one-time Yale folder rename (--undo reverses it)
  README.txt                   how to get the data and run everything

fisherfaces_project/           working directory: sources, logs, archive/
q1_matlab/                     the HW3 PCA baseline this project builds on
```

The databases are not in this repository. Extended Yale B's terms forbid
redistributing its images inside a larger publicly distributed database, so
download both from the original sources:
[ORL](https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html) and
[Extended Yale B, cropped](https://datarepo.eng.ucsd.edu/visiondata/iskwak/public_html/ExtYaleDatabase/ExtYaleB.html).
Put them one level above `final_submission/`, then run `./rename_yale.sh` once.

## Results

Recognition rate (%), 1-nearest-neighbour. Fisherfaces always uses exactly
`c−1` features, so the matched-dimension Eigenfaces column is the fair comparison.

| Dataset | Correlation | Eigenfaces (best k) | Eigenfaces @ k=c−1 | Eigenfaces w/o top 3 | **Fisherfaces** |
|---|---|---|---|---|---|
| ORL-32 (HW3 split) | 91.4 | 95.3 | 94.5 | 87.5 | 90.6 |
| ORL-40 (HW3 split) | 91.2 | 93.8 | 94.4 | 86.9 | 84.4 |
| ORL-32 (numeric split) | 93.0 | 95.3 | 95.3 | 89.1 | 93.0 |
| Yale-38 (HW3 split) | 43.0 | 34.2 | 29.5 | 58.7 | **67.2** |
| Yale-38 (random split) | 85.2 | 70.0 | 50.2 | 89.0 | **97.1** |

**The paper's claim reproduces on Yale**, where the nuisance is illumination:
Fisherfaces nearly doubles Eigenfaces (67.2 vs 34.2), and on a random split cuts
the error rate from 30.0% to 2.9%.

**It fails on ORL**, where variation is pose and expression and there are only 6
images per subject. This is reported rather than tuned away: `rank(S_W) ≤ N−c`
means each class contributes only `nᵢ−1 = 5` directions to the within-class
scatter, so `S_W` is badly estimated and LDA then inverts it. Yale has 39
directions per class and wins comfortably.

### Three findings beyond the paper

1. **`m = N−c` is the weakest part of the recipe.** It is the largest PCA
   dimension that makes `S_W` invertible, but also the *worst-conditioned* such
   choice — at `m = N−c` the rank equals the dimension exactly, with no slack, so
   the smallest eigenvalues are estimation noise and `S_W⁻¹` amplifies them.
   Sweeping `m` beats the paper's choice on both databases (ORL 95.3 vs 90.6 at
   m=60; Yale 97.0 vs 95.7 at m=1000), and `cond(S_W)` rises from 2.6e6 to 2.2e11
   over the last 282 dimensions. On one split it does not even deliver the
   invertibility it was chosen for.

2. **The lexicographic Yale split is badly biased.** `dir` sorts alphabetically
   and `'+' < '-'`, so taking "the first 40 files" trains on all 35
   positive-azimuth lightings and tests on the 24 most extreme negative ones.
   Randomising the split roughly doubles *every* method. The method *ranking* is
   unchanged, so the paper's claim is unaffected.

3. **A duplicated image was detectable from the rank alone.** `rank(S_W)`
   measured 1481 against the theoretical bound `N−c = 1482`. The cause was a stray
   byte-identical file in `CroppedYale/yaleB12/`; two identical columns stay
   identical after within-class centring, costing exactly one unit of rank.

---

## Repository layout

```
fisherfaces_project/
  toy.m                        the 2-D toy problem: why PCA is not a classifier
  eigenfaces_vs_fisherfaces.m  the main comparison, on ORL and Yale
  experiments.m                experiments E1 to E7
  figures/                     everything the three scripts draw
  report.pdf / report.tex      the 5-page project report
  study_guide.pdf / .tex       full derivations, theorems and a question bank
  run_log_*.txt                the output every quoted number comes from
  orl_glasses_labels.txt       glasses labels for ORL, made by inspection
  rename_yale.sh               one-time Yale folder rename (--undo reverses it)
  archive/                     superseded code, kept only for reproducibility
q1_matlab/
  q1_face_pca.m                the HW3 PCA baseline this project builds on
```

The databases are not in this repository. Extended Yale B's terms forbid
redistributing its images inside a larger publicly distributed database, so
download both from the original sources:
[ORL](https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html) and
[Extended Yale B, cropped](https://datarepo.eng.ucsd.edu/visiondata/iskwak/public_html/ExtYaleDatabase/ExtYaleB.html).
Put them one level above `fisherfaces_project/`, then run `./rename_yale.sh`
once.

## Results

| | correlation | eigenfaces (best k) | eigenfaces (k = c-1) | **Fisherfaces** |
|---|---|---|---|---|
| ORL (pose, expression) | 91.4% | **95.3%** | 94.5% | 90.6% |
| Yale (illumination) | 43.0% | 34.2% | 29.5% | **67.2%** |

Fisherfaces win on Yale, which is the paper's claim. They lose on ORL, which
is a real result and is explained in the report.

### The study guide

`study_guide.pdf` is the main document. It derives everything from first
principles with no skipped algebra:

- **Part I** — PCA three ways (variance / reconstruction / SVD), the `L = AᵀA`
  theorem with proof, the reconstruction-error identity, and the Lambertian
  argument for why one face under all lightings spans a 3-D subspace.
- **Part II** — Fisher's criterion, the two-class closed form
  `w ∝ S_W⁻¹(μ₁−μ₂)`, the multi-class generalised eigenproblem, `S_T = S_B + S_W`,
  and the rank theorems giving the `c−1` ceiling.
- **Part III** — the small-sample-size problem, the two-stage algorithm, Cholesky
  whitening, and the alternatives (ridge / null-space / direct LDA) with reasons.
- **Part IV** — a MATLAB syntax reference, since the code must be defended line by
  line.
- **Part V** — all experiments and measured results.
- **Part VI** — a question bank with worked answers.

---

## Running it

Requires MATLAB (developed on R2026a; base MATLAB only, no toolboxes).

The two face databases are **not** included here — they are third-party research
datasets (~80 MB) with their own terms of use. Place them at the repository root
so that the relative paths `../ORL/` and `../CroppedYale/` resolve:

```
<repo root>/
  ORL/           s1 … s40, 10 PGM images each, 92×112
  CroppedYale/   yaleB01 … yaleB39 (yaleB14 absent), 59–64 PGM images each, 168×192
  fisherfaces_project/
```

Then:

```bash
cd fisherfaces_project
matlab -batch "run('fisherfaces_simple.m')"      # ~3 minutes, writes run_log_simple.txt and figs_simple/
matlab -batch "run('fisherfaces_project.m')"     # ~8 minutes, writes run_log.txt and figs/
pdflatex study_guide.tex && pdflatex study_guide.tex
```

### Where are the figures?

Both scripts **save** their figures as `.png` and do not display them, because
`matlab -batch` has no display. Look in `fisherfaces_project/figs_simple/` and
`fisherfaces_project/figs_experiments/` (6 files each). To make the windows
open instead, set `showFigures = true` near the top of either script and run it
from the MATLAB desktop rather than with `-batch`.

### Which script to read

`fisherfaces_simple.m` is the one to read first. It is self-contained and has
three parts: a 2-D toy problem showing why PCA is not a classifier, then
eigenfaces vs Fisherfaces on ORL and Yale, then the figures. It reproduces the
same recognition rates as the full script:

| | correlation | eig (best k) | eig (k = c-1) | **Fisherfaces** |
|---|---|---|---|---|
| ORL  | 91.4% | 95.3% | 94.5% | **90.6%** |
| Yale | 43.0% | 34.2% | 29.5% | **67.2%** |

`fisherfaces_project.m` adds everything else: the sweeps over `m` and over
images per subject, six different ways of solving the generalised eigenproblem,
the alternatives to PCA-then-LDA, a frequency analysis of the two bases, and the
numerical checks.

The script is deterministic (`rng(0)`): three consecutive runs produced identical
numbers.

### Verification built into the script

| Check | Section | Result |
|---|---|---|
| HW3 eigenfaces numbers reproduced | S3/S4 | exact to all printed digits |
| `E_k = Σ_{i>k} λᵢ` (reconstruction identity) | S3/S4 | agrees to 4 decimals |
| `S_T = S_B + S_W` | S5–S8 | ≤ 8.6e-16 relative |
| `rank(S_W) ≤ N−c` | S5 | 1482 = 1482, bound met |
| our solver vs `eig(S_B,S_W,'chol')` | S6–S8 | ≤ 7.8e-16 relative |
| eigenfaces orthonormal | S12(a) | ‖UᵀU − I‖_F = 9.1e-14 |
| `eig` route vs `svd` route | S12(b) | \|uᵀu'\| = 1.000000 for top 10 |
| Fisherfaces are `S_W`-orthogonal, **not** Euclidean-orthogonal | S12(c) | 3.8e-13 vs 0.499 |
| `rank(S_B) = c−1` | S12(d) | 31 = 31 |
| closed-form `w` maximises `J` | S1 | beats a 6283-direction grid search |
| LDA error vs theoretical Bayes error | S1 | 1.70% vs 2.28% |

---

## Datasets

- **ORL / AT&T** — 40 subjects × 10 images, pose and expression variation.
- **Cropped Extended Yale B** — 38 subjects, 59–64 images each, extreme
  illumination variation. Note the per-subject count is *not* constant.

Both are used for research purposes under their original terms; neither is
redistributed here.
