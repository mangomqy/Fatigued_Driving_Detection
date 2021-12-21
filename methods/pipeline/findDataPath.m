function [dataPath] = findDataPath(sessionDataPath)

sessionFileNameList = dir(sessionDataPath);
sessionNum = length(sessionFileNameList)-2;


% 数据读取路径
for sessionIndex = 1:sessionNum
    
%     fprintf(['processing session ',num2str(sessionIndex),'\n']);
    
    
    %% load raw data
    
    sessionFileName = sessionFileNameList(2+sessionIndex).name;
    
    sessionFilePath = [sessionDataPath,'/',sessionFileName,'/EEG'];
    blockFileNameList = dir(sessionFilePath);
    blockNum = length(blockFileNameList)-2;
    
    for blockIndex = 1:blockNum
        blockFilePath  = [sessionFilePath,'/',blockFileNameList(2+blockIndex).name,'/'];
        
        blockdir = dir([blockFilePath,'*.bdf']);
        blockNames = {blockdir.name};
        
        dataPath{sessionIndex}{blockIndex}.blockNames = blockNames;
        dataPath{sessionIndex}{blockIndex}.blockFilePath = blockFilePath;
        dataPath{sessionIndex}{blockIndex}.timePeriod = sessionFileName;
        
        %         EEGOUT = readbdfdata(blockNames,blockFilePath);
    end
end
end

