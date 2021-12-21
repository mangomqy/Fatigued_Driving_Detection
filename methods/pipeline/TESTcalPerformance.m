function performance = TESTcalPerformance(eventInfo, srate, varargin)
% input:
% eventInfo: 所有事件结构体
% srate
% varargin{1}: allEvent, filteredEvent

% output: performance 行为学数据结构体(原始数据)
%   deviatime respontime：（结构：事件trial*1数组）事件时间点
%   localRT: （结构：事件trial*1数组）反应时
configData_v1;



% 20201220 version:
% session 23 24出现localRTT>10s情况，增加限制，去除异常localRT>10s trial数据

% eventInfo = EEG.event;
% srate = EEG.srate;


%% 提取local rt
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
                
                performance.respontime(trialNum) = eventInfo(eventindex).latency/srate; % 开始操作
                
                localRT = performance.respontime(trialNum) - performance.deviatime(trialNum);
                
                if localRT <= 10
                    performance.localRT(trialNum) = localRT;
                else % 反应时超出10s，异常，可能丢失trigger 250
                    fprintf(['反应时超出10s，数据可能异常，丢失此trial',num2str(trialNum),'...\n'])
                    trialNum = trialNum-1;
                end
            end
        end
        
        
    end
    
elseif strcmp(varargin{1},'allEvent')
        % 所有250都考虑，若没有按键则localRT为10

        trialNum = 0;
        
        for eventindex = 1:length(eventInfo)
            if strcmp(eventInfo(eventindex).type, '250') 
                trialNum = trialNum+1;
                
                performance.deviatime(trialNum) =  eventInfo(eventindex).latency/srate; % onset
                
                % 处理没有按键的情况，localRT = 10;
                if eventindex == length(eventInfo)
                    performance.localRT(trialNum) = 10;
                elseif strcmp(eventInfo(eventindex+1).type, '250') 
                    performance.localRT(trialNum) = 10;
                end
                
                % 检查interval
%                 if trialNum>1
%                     interval(trialNum) = performance.deviatime(trialNum)-performance.deviatime(trialNum-1);
%                     if interval(trialNum) >15
%                         s=1;
%                     end
%                 end
                 
                    
            elseif (strcmp(eventInfo(eventindex).type, '252') || strcmp(eventInfo(eventindex).type, '253'))  && eventindex > 1
                
                if strcmp(eventInfo(eventindex-1).type, '250')
                    performance.respontime(trialNum) = eventInfo(eventindex).latency/srate; % 开始操作
                    localRT = performance.respontime(trialNum) - performance.deviatime(trialNum);
                    
                    performance.localRT(trialNum) = localRT;

                    
                end
                
            end
            
        end

end



% try
%% 计算global rt

% for trialIndex = 1:trialNum
%     trialdeviatimeSet = [max(0, performance.deviatime(trialIndex)- globalwin),  ...
%         performance.deviatime(trialIndex)];
%
%     trialSet = find(performance.deviatime >= trialdeviatimeSet(1) & ...
%         performance.deviatime <= trialdeviatimeSet(2)); % 寻找在当前事件发生90s内的trial编号
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

