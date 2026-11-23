function [orlFolders, yaleFolders, dORL, dYALE, figDir] = saumya_folders()
% SAUMYA_FOLDERS  Where the data is, and how big the images are.
%
%   Kept in one place so the three scripts cannot disagree about it.
%
%   Paths resolve from THIS FILE, not from whatever folder you happen to be
%   in.  Without that the scripts only work from the project folder, and they
%   fail SILENTLY, because dir() on a missing folder returns an empty list
%   rather than an error.

thisDir = fileparts(mfilename('fullpath'));
if isempty(thisDir)
    thisDir = pwd;
end

orlRoot  = fullfile(thisDir, '..', 'ORL', filesep);
yaleRoot = fullfile(thisDir, '..', 'CroppedYale', filesep);
figDir   = fullfile(thisDir, 'figs_saumya', filesep);

if exist(orlRoot, 'dir') == 0
    error('Cannot find ORL at %s', orlRoot);
end
if exist(yaleRoot, 'dir') == 0
    error('Cannot find CroppedYale at %s', yaleRoot);
end
if exist(figDir, 'dir') == 0
    mkdir(figDir);
end

% ORL: build the path from the index.  Do NOT glob 's*' -- this copy of ORL
% has a stray empty folder called 'sed' that a glob would pick up.
orlFolders = cell(1, 32);
for p = 1:32
    orlFolders{p} = [orlRoot 's' num2str(p) '/'];
end

% Yale: same loop.  The folders shipped as yaleB01..yaleB39 with yaleB14
% missing, which is not a countable range, so rename_yale.sh renamed them to
% s1..s38.  Safe, because the label is the loop index, not the folder name.
yaleFolders = cell(1, 38);
for p = 1:38
    yaleFolders{p} = [yaleRoot 's' num2str(p) '/'];
end

dORL  = 112 * 92;      % imread gives rows x cols, so 112 rows and 92 columns
dYALE = 192 * 168;
end
