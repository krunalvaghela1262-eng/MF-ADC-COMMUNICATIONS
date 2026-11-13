function setup_paths()
%SETUP_PATHS  Add every folder of this repository to the MATLAB path.
%   Run once per MATLAB session from the repository root:  >> setup_paths

    root = fileparts(mfilename('fullpath'));
    addpath(genpath(fullfile(root, 'src')));
    addpath(fullfile(root, 'experiments'));
    fprintf('Modulo-ADC symbol detection: paths added from %s\n', root);
end
