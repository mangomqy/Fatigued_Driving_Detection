function globaldata = movingwinForGlobalRT(localdata,deviatime,globalwin,type)
% ��local rt /di ƽ��


if strcmp(type,'rt')
    
    trialNum = length(localdata);


    for trialIndex = 1:trialNum
        trialdeviatimeSet = [max(0, deviatime(trialIndex)- globalwin),  ...
            deviatime(trialIndex)];
        
        trialSet = find(deviatime >= trialdeviatimeSet(1) & ...
            deviatime <= trialdeviatimeSet(2)); % Ѱ���ڵ�ǰ�¼�����90s�ڵ�trial��ţ�9���¼�
        
        if length(trialSet)<9 %�ų�10sһ���¼�������¼�����С��8����˵�����ܷ����¹ʣ���ǰ��������ƽ��9��
            
            trialSet = [max(1,trialSet(end)-8):trialSet(end)];
            
            if length(trialSet)<9
                
                trialSet = [trialSet(1):trialSet(1)+8]; % ǰ�����¼�����ƽ��
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
            deviatime <= trialdeviatimeSet(2)); % Ѱ���ڵ�ǰ�¼�����90s�ڵ�trial��ţ�9���¼�
        
        if length(trialSet)<9 %�ų�10sһ���¼�������¼�����С��8����˵�����ܷ����¹ʣ���ǰ��������ƽ��9��
            
            trialSet = [max(1,trialSet(end)-8):trialSet(end)];
            
            if length(trialSet)<9
                
                trialSet = [trialSet(1):trialSet(1)+8]; % ǰ�����¼�����ƽ��
            end
            
        end
        
        globaldata(:,:,trialIndex) = mean(localdata(:,:,trialSet),3);
        
    end
    
    
end

