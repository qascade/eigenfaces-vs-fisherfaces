# Eigenfaces vs Fisherfaces

CS 740 course project: a MATLAB implementation of Belhumeur, Hespanha &
Kriegman, *"Eigenfaces vs. Fisherfaces: Recognition Using Class Specific Linear
Projection"*, IEEE TPAMI 19(7), 1997, with seven experiments of our own.

## Results

| | correlation | eigen (best k) | eigen (k = c-1) | eigen w/o top 3 | **Fisherfaces** |
|---|---|---|---|---|---|
| ORL (pose, expression) | 91.41% | **95.31%** | 94.53% | 87.50% | 90.62% |
| Yale (illumination) | 42.95% | 34.23% | 29.53% | 58.72% | **67.23%** |

Fisherfaces win on Yale, which is the paper's claim. They lose on ORL, which is
a real result and is explained in the report. Dropping the top three components
helps Yale and hurts ORL -- the same argument seen from both sides.

## Layout

```
sk/                            the submission
  toy.m                        the 2-D problem: why PCA is not a classifier
  eigenfaces_vs_fisherfaces.m  the four-method comparison on ORL and Yale
  experiments.m                experiments E1 to E7
  figures/                     the eleven figures the scripts draw
  report.pdf / report.tex      the project report
  study_guide.pdf / .tex       viva preparation on the submitted code
  run_log_*.txt                the output every quoted number comes from
  orl_glasses_labels.txt       glasses labels for ORL, made by inspection
  rename_yale.sh               one-time Yale folder rename (--undo reverses it)
  reference_code/              the earlier code this was built from
  README.txt                   how to get the data and run everything

q1_matlab/                     the HW3 PCA baseline this project builds on
```

## The experiments

| | question |
|---|---|
| E1 | Is the paper's `m = N-c` the best choice? (No, on either database.) |
| E2 | Is the result real, or is the pipeline leaking labels? |
| E3 | Can `m` be chosen without looking at the test set? |
| E4 | Are the higher-frequency Fisherfaces more fragile under noise? |
| E5 | How does error grow with the angle of the light source? |
| E6 | Does the `c-1` ceiling on the number of Fisherfaces really bind? |
| E7 | Glasses or no glasses: the paper's section 3.3. |

## Data

Not in this repository. Extended Yale B's terms forbid redistributing its
images inside a larger publicly distributed database, so download both from the
original sources:
[ORL](https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html) and
[Extended Yale B, cropped](https://datarepo.eng.ucsd.edu/visiondata/iskwak/public_html/ExtYaleDatabase/ExtYaleB.html).
Put them one level above `sk/`, then run `./rename_yale.sh` once. See
`sk/README.txt` for the full setup.
