%% datapath
fileFolder = pwd; % filesep
EEGsessionDataPath =[fileFolder,'/data/rawEEG-bdf']; % raw .bdf data
EEGsessionDataPath_mat = [fileFolder,'/data/rawEEG-mat']; % transfered .mat data
EEGsessionDataPath_preprocessed = [fileFolder,'/data/preprocessedEEG-mat']; % preprocessed data
EEGperformDataPath = [fileFolder,'/data/performData'];
EEGepochDataPath = [fileFolder ,'/data/EpochData'];
EEGdataPath = findDataPath(EEGsessionDataPath);
sessionNum = length(EEGdataPath);

% channel loc 
locPath63 = [pwd,'/methods/automagic-master/matlab_scripts/neuracle_64_yxy.loc'];
% for performance data
performanceDataPath = [fileFolder,'/data/performanceData'];

featureLabelDataPath = [fileFolder, '/data/featureLabelData'];

% for eeg preprocess
name = 'fatigue-driving-2020-automagic';
dataFolder = [EEGsessionDataPath_mat];
resultsFolder = [EEGsessionDataPath_preprocessed];




% for feature + label + performancert data 存下来的数据矩阵的参数 
featureLabelDataPara = {'allEvent', ... allEvent or removeOutlier or ksigma 
    'alltrial',...% alltrial or 2class
    'sep', ...% mixd or sep
    'both', ...% classification or corranalysis or both
    'nocsp' ...% csp or nocsp
    };


%% sub info

subNumSet = [1];

subSet = {[1]};

sessionSet = cat(2,subSet{:});

for i = 1:sessionNum
    subSetFileName{i} = EEGdataPath{i}{1}.timePeriod; 
end


%% eeg processing parameters
globalwin = 60*1.5; 
ksigma = 3;
frange = [1,30]; 
threshold_rt = 0.15;
dist_trainx_center = 'euclidean';

%% 通道相关

chanNum = 63;
chan30 = [6,7,53,54,61:63];

%% 2class
className = {'alert','drowsy'};
classNum = length(className);