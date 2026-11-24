EIGENFACES vs FISHERFACES -- CS 740
Submission: code, report, study guide, figures.  Databases are NOT included.

SUBMITTED CODE  (base MATLAB only, no toolboxes)
  toy.m                        the 2-D problem: why PCA is not a classifier
  eigenfaces_vs_fisherfaces.m  the four-method comparison on ORL and Yale
  experiments.m                experiments E1 to E4
  figures/                     the eight figures the scripts draw

DOCUMENTS
  report.pdf        the project report      (report.tex is the source)
  study_guide.pdf   viva preparation: theory behind the submitted code and
                    experiments, plus questions and answers
  run_log_*.txt     the output every number in both documents comes from

SUPPORTING
  rename_yale.sh           one-time Yale folder rename, see below
  reference_code/          earlier code this submission was built from:
                             hw3_baseline_pca.m                 the HW3 PCA baseline
                             lda_toy_minimal_saumya.m           first toy version
                             eigenfaces_fisherfaces_simple_saumya.m
                                                                first faces version

GET THE DATA
  ORL   https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html
  Yale  https://datarepo.eng.ucsd.edu/visiondata/iskwak/public_html/ExtYaleDatabase/ExtYaleB.html

  Not bundled: Extended Yale B's terms forbid redistributing its images inside
  a larger publicly distributed database.

SET UP
  Put both databases one level ABOVE this folder:

      project/
        ORL/            40 folders s1..s40, 10 PGM images each
        CroppedYale/    38 folders, about 64 PGM images each
        sk/             these files

  CroppedYale ships as yaleB01..yaleB39 with yaleB14 missing, which is not a
  countable range, so run once:

      ./rename_yale.sh            (./rename_yale.sh --undo reverses it)

  No result changes: the class label is the loop index, never the folder name.

RUN
      matlab -batch "run('toy.m')"                        seconds
      matlab -batch "run('eigenfaces_vs_fisherfaces.m')"  about 2 minutes
      matlab -batch "run('experiments.m')"                about 10 minutes

WHAT TO EXPECT
                   correlation  eig(best k)  eig(k=c-1)  eig w/o top3  Fisherfaces
      ORL              91.41       95.31       94.53        87.50         90.62
      Yale             42.95       34.23       29.53        58.72         67.23

  Fisherfaces win on Yale, which is the paper's claim.  They lose on ORL, which
  is a real result and is explained in the report.
