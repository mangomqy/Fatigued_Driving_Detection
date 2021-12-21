function globaldata = movingwinForGlobalRT(localdata,deviatime,globalwin,type)
% 对local rt /di 平滑


if strcmp(type,'rt')
    
    trialNum = length(localdata);


    for trialIndex = 1:trialNum
        trialdeviatimeSet = [max(0, deviatime(trialIndex)- globalwin),  ...
            deviatime(trialIndex)];
        
        trialSet = find(deviatime >= trialdeviatimeSet(1) & ...
            deviatime <= trialdeviatimeSet(2)); % 寻找在当前事件发生90s内的trial编号：9个事件
        
        if length(trialSet)<9 %张长10s一个事件，如果事件数量小于8个，说明可能发生事故，往前倒，至少平均9个
            
            trialSet = [max(1,trialSet(end)-8):trialSet(end)];
            
            if length(trialSet)<9
                
                trialSet = [trialSet(1):trialSet(1)+8]; % 前几个事件往后平均
            end
            
        end
        
        globaldata(trialIndex) = mean(localdata(trialSet));
        
    end

elseif strcmp(type,'eeg')
    
    trialNum = size(localdata,3);


    for trialIndex = 1:trialNum
        trialdeviatimeSet = [max(0, deviatime(trialIndex)- globalwin),  ...
            deviatime(trialIndex)];
        
        trialSet = find(deviatime >= trialdeviatimeSet(1) & ...
            deviatime <= trialdeviatimeSet(2)); % 寻找在当前事件发生90s内的trial编号：9个事件
        
        if length(trialSet)<9 %张长10s一个事件，如果事件数量小于8个，说明可能发生事故，往前倒，至少平均9个
            
            trialSet = [max(1,trialSet(end)-8):trialSet(end)];
            
            if length(trialSet)<9
                
                trialSet = [trialSet(1):trialSet(1)+8]; % 前几个事件往后平均
            end
            
        end
        
        globaldata(:,:,trialIndex) = mean(localdata(:,:,trialSet),3);
        
    end
    
    
end

