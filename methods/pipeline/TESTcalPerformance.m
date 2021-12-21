function performance = TESTcalPerformance(eventInfo, srate, varargin)
% input:
% eventInfo: �����¼��ṹ��
% srate
% varargin{1}: allEvent, filteredEvent

% output: performance ��Ϊѧ���ݽṹ��(ԭʼ����)
%   deviatime respontime�����ṹ���¼�trial*1���飩�¼�ʱ���
%   localRT: ���ṹ���¼�trial*1���飩��Ӧʱ
configData_v1;



% 20201220 version:
% session 23 24����localRTT>10s������������ƣ�ȥ���쳣localRT>10s trial����

% eventInfo = EEG.event;
% srate = EEG.srate;


%% ��ȡlocal rt
if strcmp(varargin{1},'filteredEvent')
    trialNum = 0;
    tempcount = 0;
    
    
    for eventindex = 1:length(eventInfo)
        
        
        if strcmp(eventInfo(eventindex).type, '250')  && eventindex < length(eventInfo)
            
            if strcmp(eventInfo(eventindex+1).type, '252') || strcmp(eventInfo(eventindex+1).type, '253')
                trialNum = trialNum+1;
                
                performance.deviatime(trialNum) =  eventInfo(eventindex).latency/srate; % onset
            end
            
            if strcmp(eventInfo(eventindex+1).type, '250')
                tempcount = tempcount +1;
            end
            
        elseif (strcmp(eventInfo(eventindex).type, '252') || strcmp(eventInfo(eventindex).type, '253')) ...
                && eventindex > 1
            
            if strcmp(eventInfo(eventindex-1).type, '250')
                
                performance.respontime(trialNum) = eventInfo(eventindex).latency/srate; % ��ʼ����
                
                localRT = performance.respontime(trialNum) - performance.deviatime(trialNum);
                
                if localRT <= 10
                    performance.localRT(trialNum) = localRT;
                else % ��Ӧʱ����10s���쳣�����ܶ�ʧtrigger 250
                    fprintf(['��Ӧʱ����10s�����ݿ����쳣����ʧ��trial',num2str(trialNum),'...\n'])
                    trialNum = trialNum-1;
                end
            end
        end
        
        
    end
    
elseif strcmp(varargin{1},'allEvent')
        % ����250�����ǣ���û�а�����localRTΪ10

        trialNum = 0;
        
        for eventindex = 1:length(eventInfo)
            if strcmp(eventInfo(eventindex).type, '250') 
                trialNum = trialNum+1;
                
                performance.deviatime(trialNum) =  eventInfo(eventindex).latency/srate; % onset
                
                % ����û�а����������localRT = 10;
                if eventindex == length(eventInfo)
                    performance.localRT(trialNum) = 10;
                elseif strcmp(eventInfo(eventindex+1).type, '250') 
                    performance.localRT(trialNum) = 10;
                end
                
                % ���interval
%                 if trialNum>1
%                     interval(trialNum) = performance.deviatime(trialNum)-performance.deviatime(trialNum-1);
%                     if interval(trialNum) >15
%                         s=1;
%                     end
%                 end
                 
                    
            elseif (strcmp(eventInfo(eventindex).type, '252') || strcmp(eventInfo(eventindex).type, '253'))  && eventindex > 1
                
                if strcmp(eventInfo(eventindex-1).type, '250')
                    performance.respontime(trialNum) = eventInfo(eventindex).latency/srate; % ��ʼ����
                    localRT = performance.respontime(trialNum) - performance.deviatime(trialNum);
                    
                    performance.localRT(trialNum) = localRT;

                    
                end
                
            end
            
        end

end



% try
%% ����global rt

% for trialIndex = 1:trialNum
%     trialdeviatimeSet = [max(0, performance.deviatime(trialIndex)- globalwin),  ...
%         performance.deviatime(trialIndex)];
%
%     trialSet = find(performance.deviatime >= trialdeviatimeSet(1) & ...
%         performance.deviatime <= trialdeviatimeSet(2)); % Ѱ���ڵ�ǰ�¼�����90s�ڵ�trial���
%
%     performance.globalRT(trialIndex) = mean(performance.localRT(trialSet));
%
%     performance.globalDI(trialIndex) = mean(performance.localDI(trialSet));
%
% end
% catch
%     fprintf('error')
%     s=1;
% end

% globaldata = movingwinForGlobalRT(performance.localRT,performance.deviatime,globalwin);
% performance.globalRT = globaldata;



end

