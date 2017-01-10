function [ SInfo, TPLStructs ] = generate_MatCells_from_sorted_data( fPlexon, fTPLStruct, is_combined )
% Loads spike sorted data into TPLCell array structure, generates trial
% parameters and saves it as TPLStruct array.
%
% Parameters:
% -----------
% fPlexon: full file path to sorted data file
% fTPLStruct: file path to save TPLStruct into
% is_combined: flag indicating task type, 0: direction or location, 1: combined [optional]
%
% Output:
% -------
% SInfo: information about a TPL data file, see TPLFileInfo
% TPLStructs: array of TPLStruct objects (TPLCell class objects converted to simple Matlab structures)
%
% David Samu, Sep 2016


%% Init
addpath(genpath('D:\DavidSamu\tools\matlab\UnitCreation'))

if nargin < 3
    is_combined = 0;
end

% Check if file exists.
if exist(fPlexon, 'file') ~= 2
    warning('File %s is not found', fPlexon)
    return
end


%% Import spike sorted data file

try
    % Create session info object.
    SInfo = TPLFileInfo(fPlexon);
    
    % Get find sorted units.
    % 1st row is unsorted spikes, 1st column is empty!
    [iunits, ichans] = find(SInfo.wfcounts(2:end,2:end));
    nunit = length(iunits);
    TPLCells = SimpleTPLCell.empty(0,0);

    % Load each unit from each channel.
    for iu = 1:nunit     
        
        iunit = iunits(iu);
        ichan = ichans(iu);
        fprintf('      %i/%i (channel %i, unit %i)\n', iu, nunit, ichan, iunit);         
        TPLCells(iu) = SimpleTPLCell(fPlexon, ichan, iunit);
        
    end    
catch
    error('Importing file %s as TPL file was unsuccessful.', fPlexon);
    return
end


%% Generate trial parameters
if is_combined
    mapD_load_trials_mixed(TPLCells);
else
    mapD_load_trials(TPLCells);
end


%% Convert TPLCell class objects into Matlab structure.
nunits = length(TPLCells);
TPLStructs = cell(1, nunits);
warning('off', 'MATLAB:structOnObject');
for icell = 1:nunits
    TPLStructs{icell} = struct(TPLCells(icell));
end
warning('on', 'MATLAB:structOnObject');


%% Save TPL structure.
if ~isempty(fTPLStruct)
    [fdir, ~, ~] = fileparts(fTPLStruct);
    if ~exist(fdir, 'dir')
       mkdir(fdir)
    end    
    save(fTPLStruct, 'SInfo', 'TPLStructs');
end 

end

