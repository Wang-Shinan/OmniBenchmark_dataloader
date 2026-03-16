% MATLAB script to batch convert ANT Neuro .cnt files to .mat format
% Usage: Run this in MATLAB with EEGLAB installed
% 
% This script converts all ANT Neuro .cnt files in a directory to .mat format
% that can be read by MNE-Python using scipy.io.loadmat
%
% Requirements:
% - MATLAB with EEGLAB toolbox
% - EEGLAB: https://sccn.ucsd.edu/eeglab/download.php

function batch_convert_ant_to_mat(cnt_dir, output_dir, pattern)
    % cnt_dir: Directory containing .cnt files
    % output_dir: Directory to save .mat files (default: same as cnt_dir)
    % pattern: File pattern to match (default: '*.cnt')
    
    if nargin < 2
        output_dir = cnt_dir;
    end
    if nargin < 3
        pattern = '*.cnt';
    end
    
    % Add EEGLAB to path if not already added
    if ~exist('eeglab', 'file')
        error('EEGLAB not found. Please add EEGLAB to MATLAB path.');
    end
    
    % Ensure output directory exists
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end
    
    % Find all .cnt files
    files = dir(fullfile(cnt_dir, pattern));
    
    if isempty(files)
        fprintf('No .cnt files found in %s\n', cnt_dir);
        return;
    end
    
    fprintf('Found %d .cnt files to convert\n', length(files));
    fprintf('Output directory: %s\n', output_dir);
    fprintf('Starting conversion...\n\n');
    
    success_count = 0;
    fail_count = 0;
    
    for i = 1:length(files)
        cnt_file = fullfile(cnt_dir, files(i).name);
        fprintf('[%d/%d] Processing: %s\n', i, length(files), files(i).name);
        
        try
            % Read ANT file using EEGLAB
            fprintf('  Reading file...\n');
            EEG = pop_loadcnt(cnt_file, 'dataformat', 'auto');
            
            fprintf('  File loaded: %d channels, %d samples, %.2f Hz\n', ...
                EEG.nbchan, EEG.pnts, EEG.srate);
            
            % Extract data and channel information
            data = EEG.data;  % channels x samples
            ch_names = {EEG.chanlocs.labels};
            sfreq = EEG.srate;
            
            % Handle missing channel names
            if isempty(ch_names) || all(cellfun(@isempty, ch_names))
                fprintf('  Warning: No channel names found, using default names\n');
                ch_names = cell(1, size(data, 1));
                for j = 1:length(ch_names)
                    ch_names{j} = sprintf('CH%d', j);
                end
            end
            
            % Create output structure
            output.data = data;
            output.ch_names = ch_names;
            output.sfreq = sfreq;
            output.n_channels = size(data, 1);
            output.n_samples = size(data, 2);
            output.original_file = files(i).name;
            
            % Save to .mat file
            [~, name, ~] = fileparts(cnt_file);
            output_file = fullfile(output_dir, [name, '.mat']);
            
            fprintf('  Saving to: %s\n', output_file);
            save(output_file, 'output', '-v7.3');
            
            fprintf('  ✓ Conversion complete!\n\n');
            success_count = success_count + 1;
            
        catch ME
            fprintf('  ✗ Error: %s\n', ME.message);
            fprintf('  Stack trace:\n');
            for k = 1:length(ME.stack)
                fprintf('    %s (line %d)\n', ME.stack(k).name, ME.stack(k).line);
            end
            fprintf('\n');
            fail_count = fail_count + 1;
        end
    end
    
    fprintf('\n========================================\n');
    fprintf('Conversion Summary:\n');
    fprintf('  Success: %d files\n', success_count);
    fprintf('  Failed:  %d files\n', fail_count);
    fprintf('  Total:   %d files\n', length(files));
    fprintf('========================================\n');
end

% Example usage:
% batch_convert_ant_to_mat('/mnt/dataset2/Datasets/SEED_FRA/French/01-EEG-raw')
% batch_convert_ant_to_mat('/mnt/dataset2/Datasets/SEED_GER/German/01-EEG-raw')

