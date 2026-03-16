% MATLAB script to convert ANT Neuro .cnt files to .mat format
% Usage: Run this in MATLAB with EEGLAB installed
% 
% This script converts ANT Neuro .cnt files to .mat format that can be
% read by MNE-Python using scipy.io.loadmat
%
% Requirements:
% - MATLAB with EEGLAB toolbox
% - EEGLAB: https://sccn.ucsd.edu/eeglab/download.php

function convert_ant_to_mat(cnt_file, output_dir)
    % cnt_file: Path to .cnt file
    % output_dir: Directory to save .mat file
    
    % Add EEGLAB to path if not already added
    if ~exist('eeglab', 'file')
        error('EEGLAB not found. Please add EEGLAB to MATLAB path.');
    end
    
    fprintf('Reading ANT file: %s\n', cnt_file);
    
    % Read ANT file using EEGLAB
    % Note: EEGLAB can read ANT Neuro files
    try
        EEG = pop_loadcnt(cnt_file, 'dataformat', 'auto');
    catch
        % Try alternative method
        EEG = pop_fileio(cnt_file);
    end
    
    fprintf('File loaded: %d channels, %d samples, %.2f Hz\n', ...
        EEG.nbchan, EEG.pnts, EEG.srate);
    
    % Extract data and channel information
    data = EEG.data;  % channels x samples
    ch_names = {EEG.chanlocs.labels};
    sfreq = EEG.srate;
    
    % Create output structure
    output.data = data;
    output.ch_names = ch_names;
    output.sfreq = sfreq;
    output.n_channels = size(data, 1);
    output.n_samples = size(data, 2);
    
    % Save to .mat file
    [~, name, ~] = fileparts(cnt_file);
    output_file = fullfile(output_dir, [name, '.mat']);
    
    fprintf('Saving to: %s\n', output_file);
    save(output_file, 'output', '-v7.3');
    
    fprintf('Conversion complete!\n');
end

