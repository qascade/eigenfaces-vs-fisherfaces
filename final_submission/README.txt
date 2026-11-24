EIGENFACES vs FISHERFACES -- CS 740 project
Final submission: code, report and figures.  The face databases are not here.

FILES
  toy.m                        the 2-D toy problem: why PCA is not a classifier
  eigenfaces_vs_fisherfaces.m  the main comparison, on ORL and Yale
  experiments.m                experiments E1 to E7
  figures/                     all eleven figures, drawn by the three scripts
  report.pdf                   the report (report.tex is its source)
  study_guide.pdf              full derivations, theorems and a question bank
  run_log_*.txt                the output every number in the report comes from
  orl_glasses_labels.txt       glasses labels for ORL, made by visual inspection
  rename_yale.sh               one-time folder rename, see below

GET THE DATA
  ORL   https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html
  Yale  https://datarepo.eng.ucsd.edu/visiondata/iskwak/public_html/ExtYaleDatabase/ExtYaleB.html

  They are not bundled because Extended Yale B's terms forbid redistributing
  its images inside a larger publicly distributed database.

SET UP
  Put both databases one level ABOVE this folder:

      project/
        ORL/                 40 folders s1..s40, 10 PGM images each
        CroppedYale/         38 folders, about 64 PGM images each
        final_submission/    these files

  CroppedYale arrives as yaleB01..yaleB39 with yaleB14 missing, which is not
  a countable range, so run this once:

      ./rename_yale.sh              (./rename_yale.sh --undo reverses it)

  It renames the folders to s1..s38 so both databases load with the same
  loop.  No result changes: the class label is the loop index, never the
  folder name.

RUN  (base MATLAB only, no toolboxes)
      matlab -batch "run('toy.m')"                        seconds
      matlab -batch "run('eigenfaces_vs_fisherfaces.m')"  about 2 minutes
      matlab -batch "run('experiments.m')"                about 10 minutes

  Each script writes its figures into figures/.

WHAT TO EXPECT
                       correlation  eig(best k)  eig(k=c-1)  Fisherfaces
      ORL   (pose)         91.4%       95.3%       94.5%        90.6%
      Yale  (lighting)     43.0%       34.2%       29.5%        67.2%

  Fisherfaces win on Yale, which is the paper's claim.  They lose on ORL,
  which is a real result and is explained in the report.
