SUPERSEDED CODE -- not part of the submission.

Kept only so the earlier results stay reproducible.  The submission is the
three scripts one level up: toy.m, eigenfaces_vs_fisherfaces.m, experiments.m.

  fisherfaces_project.m        the original 1600-line monolithic script.
                               The study guide's numbers come from this, and
                               run_log.txt one level up is its output.
  fisherfaces_simple.m         an earlier short version, replaced by
                               eigenfaces_vs_fisherfaces.m
  fisherfaces_experiments.m    an earlier experiments file, replaced by
                               experiments.m
  figs_simple/                 its figures
  run_log_simple.txt           its output

  lda_toy_minimal_saumya.m                   the two starting files these
  eigenfaces_fisherfaces_simple_saumya.m     three scripts were built from.
  Copy_of_...                                a stray duplicate.

  NOTE on eigenfaces_fisherfaces_simple_saumya.m: it has a real bug.
  eig(Sb,Sw) returns columns satisfying W'*S_W*W = I, NOT unit Euclidean
  length -- here they differ by a factor of 287.  Nearest neighbour with
  Euclidean distance is not scale invariant, so the long directions dominate
  and Fisherfaces score 29.50% instead of 88.50%.  Fixed in
  eigenfaces_vs_fisherfaces.m by normalising each column.
