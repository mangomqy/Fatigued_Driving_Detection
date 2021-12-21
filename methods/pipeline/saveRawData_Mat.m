function saveRawData_Mat

configData_v1;

% chanlocs = readlocs(locPath63);



for sessionIndex = 1:sessionNum
    
    fprintf(['processing session ',num2str(sessionIndex),'\n']);
    
    blockNum = length(EEGdataPath{sessionIndex});
    
    
    
    for blockIndex = 1:blockNum
        
        subName = EEGdataPath{sessionIndex}{blockIndex}.timePeriod(1:3);
        %         sub = str2num(EEGdataPath{sessionIndex}{blockIndex}.timePeriod(2:3));
        reshapeDataName = EEGdataPath{sessionIndex}{blockIndex}.timePeriod;
        
        
        
        saveFile = [EEGsessionDataPath_mat,'/',subName,'/',reshapeDataName,'-EEG-',num2str(blockIndex),'.mat'];
        preprocessedfile = [EEGsessionDataPath_preprocessed,'/',subName,'/','nip_',reshapeDataName,'-EEG-',num2str(blockIndex),'.mat']
        if ~exist(saveFile,'file')
            if ~isfolder([EEGsessionDataPath_mat,'/',subName])
                
                mkdir(EEGsessionDataPath_mat,subName)
                
            end
            
            EEG = readbdfdata(EEGdataPath{sessionIndex}{blockIndex}.blockNames,...
                EEGdataPath{sessionIndex}{blockIndex}.blockFilePath);
            
%             EEG.chanlocs = chanlocs;
            
            save(saveFile,'EEG')
        else
            fprintf([reshapeDataName,' is already existing...skipping...\n'])
        end
       %%
        if ~exist(preprocessedfile,'file')
            if ~isfolder([EEGsessionDataPath_preprocessed,'/',subName])
                
                mkdir(EEGsessionDataPath_preprocessed,subName)
                
            end
            % filter with 1-100Hz
            [EEG, ~, ~] = pop_eegfiltnew(EEG,1,45);
            
            % downsample to 250Hz
            EEG = pop_resample(EEG, 250);
            
            save(preprocessedfile,'EEG')
        else
            fprintf([reshapeDataName,'processed_data is already existing...skipping...\n'])
        end

       
    end
    
    
end

end

