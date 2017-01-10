%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Script to generate TPLCell structure from spike sorted Plexon files
%
% David Samu
% Sep 2016
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Init.
addpath(genpath(strcat('Y:\yesbackup\Users\DavidSamu\tools\matlab')))


%% Which recordings to process?

% Here you specify the path to the recordings to be processed.
base_dir = 'Y:\yesbackup\ElectrophysiologyData\Plexon';
monkey = '201';
subdir = 'PFC';
dates = {'062416', '071316', '071516'};

% Set this to 1, if experiment is Combined location + direction, otherwise 0.
is_combined = 1;


%% Rename split Plexon files (after spike sorting and splitting merged recording data file).

fprintf('\n\nRenaming sorted tasks\n');

for i = 1:length(dates)
    date = dates{i};
    recording = sprintf('%s_%s', monkey, date);
    
    fprintf('  %s\n', recording);

    % Init folders.
    recording_dir = sprintf('%s\\%s\\%s\\%s\\', base_dir, monkey, subdir, recording);
    unsorted_dir = strcat(recording_dir, 'Unsorted\');
    sorted_dir = strcat(recording_dir, 'Sorted\Split\');

    % Init files. This will only return files with extension '.plx'.
    unsorted_files = dir(strcat(unsorted_dir, '*.plx'));
    sorted_files = dir(strcat(sorted_dir, '*.plx'));  

    % Some error checking. Recording is skipped if any error found.
    skip_rec = 0;  % set to 1 if error found
    % Unequal number of Plexon files.
    nunsorted = length(unsorted_files);
    nsorted = length(sorted_files);    
    if nunsorted ~= nsorted
        warning('Unequal number of Plexon files in unsorted and sorted folders.')
    end
    % Check if file naming follows expected pattern.
    for files_cell = {unsorted_files, sorted_files}
        files = files_cell{1};
        for j = 1:length(files)        
            file = files(j);
            
            % Init parts of file name.
            [path, name, ext] = fileparts(file.name);
            name_parts = strsplit(name, '_');
            
            % Insufficient number of file parts.
            if (length(name_parts) ~= 4) 
                warning('%s could not be split into exactly 4 parts.', name)
                skip_rec = 1;
            end
            % Monkey name.
            if ~strcmp(monkey, name_parts{1})
                warning('Folder and file monkey names do not match ("%s" and "%s").', monkey, name_parts{1})
                skip_rec = 1;
            end
            % Recording date.
            fname_date = name_parts{2}(1:6);
            if ~strcmp(date, fname_date)
                warning('Folder and file dates do not match ("%s" and "%s").', date, fname_date)
                skip_rec = 1;
            end
        end
    end
    % If there is any inconsistency, skip processing recording.
    if skip_rec
         warning('Skipping recording! Please fix indicated naming issues before creating TPLCells below!');
         continue
    end
             
    % At this point, we have the same number of plexon files in both
    % folders, and their naming satisfies all the above criteria.
    
    % Go through original task files from unsorted folder and find
    % corresponding sorted files.
    file_pairs = {};    
    for j = 1:length(unsorted_files)        
        unsorted_file = unsorted_files(j);
        
        % Split file name into subparts.
        % This should result in: [monkey, date+electrode, taskname, '0']        
        [path, unsorted_name, ext] = fileparts(unsorted_file.name);        
        unsorted_name_parts = strsplit(unsorted_name, '_'); 
                 
        % Find matching split and sorted plexon file.
        for k = 1:length(sorted_files)
            sorted_file = sorted_files(k);  
            
            % Init sorted split file.
            % This should result in: [monkey, date+electrode, 'spl', 'f00[idx]']    
            [path, sorted_name, ext] = fileparts(sorted_file.name);     
            sorted_name_parts = strsplit(sorted_name, '_');
           
            % Find file by name and index. 
            monkey_match = strcmp(unsorted_name_parts{1}, sorted_name_parts{1});
            date_elec_match = strcmp(unsorted_name_parts{2}, sorted_name_parts{2});            
            index_match = strcmp(unsorted_name_parts{3}(end), sorted_name_parts{4}(end));
            
            if monkey_match && date_elec_match && index_match

                % Check if file size matches.
                if unsorted_file.bytes ~= sorted_file.bytes
                    warning('File sizes of sorted and unsorted plexon files do not match. Merging was done with wrong task order?');
                    continue
                end
                
                % Save pair after all tests passed.
                file_pairs{end+1} = {sorted_file.name unsorted_file.name};
                continue
            end
        end
    end
    
    % Skip recording if not all files have pairs.
    if length(file_pairs) ~= length(unsorted_files)
         warning('Not found matching pair for all Plexon files. Skipping recording!');
         continue
    end
    
    % Rename sorted split files to their unsorted file names.
    for j = 1:length(file_pairs)
        sorted_fname = file_pairs{j}{1};
        unsorted_fname = file_pairs{j}{2};
        orig_pwd = pwd; 
        cd(sorted_dir);
        movefile(sorted_fname, unsorted_fname);
        cd(orig_pwd);
    end
end

    
%% Load spike sorted data and save into TPLCell and SealCell structure.

fprintf('\n\nCreating TPLCell and SealCell structures\n');

for i = 1:length(dates)
    date = dates{i};
    recording = sprintf('%s_%s', monkey, date);
    
    fprintf('  %s\n', recording);
    
    % Init folder and query all tasks. Every (plexon) file in folder 
    % is assumed to be a task file to be processed!
    rec_dir = sprintf('%s\\%s\\%s\\%s\\', base_dir, monkey, subdir, recording);
    split_dir = strcat(rec_dir, 'Sorted\Split\');   % folder with sorted split plexon files
    TPL_dir = strcat(rec_dir, 'TPLCell\');   % output directory
    task_files = dir(strcat(split_dir, '*.plx'));
    
    % Process each task.
    for j = 1:length(task_files)       
        task_fname = task_files(j).name;
        
        fprintf('    %s\n', task_fname);        

        % Init input (.plx) and output (.mat) file paths.
        fPlexon = strcat(split_dir, task_fname);
        [path, sorted_name, ext] = fileparts(task_fname);
        fTPLCell = strcat(TPL_dir, sorted_name, '.mat');
        
        % Generate TPLCells. This takes a while...
        generate_MatCells_from_sorted_data( fPlexon, fTPLCell, is_combined );
        
    end
    
end
