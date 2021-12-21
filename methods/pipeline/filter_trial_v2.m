function [fea_output,target_output,timestamp_output] = filter_trial_v2(fea_alltrial,performanceData_alltrial,...
    trialSelectMode,classficationMode,featureFilterMode,targetName)
% 根据输入过滤trial
% input: 
    % fea_alltrial{session}{block}{channel}.struct: 
    % performanceData_alltrial{session} {block}
    
    % trialSelectMode:  allEvent removeOutlier ksigma 
    % classficationMode: alltrial 2class 
    
    % featureFilterMode: classification  corranalysis []
    % targetName: 
% output:
    % fea_output:
     
configData_v1;

%% 计算行为学数据
for sessionIndex = 1:length(cat(2,subSet{:})) 

    
    performanceData = performanceData_alltrial{sessionIndex};
    
    blcbkNum = length(performanceData);
    
    for blockIndex= 1:blcbkNum
    %% trialSelectMode
     
        
    if strcmp(trialSelectMode,'ksigma')
        trialIndex_removeOutlier = intersect( ...
            find(performanceData{blockIndex}.localRT > threshold_rt), ...
            find(performanceData{blockIndex}.localRT < 10));
        
        rt_mean = mean(performanceData{blockIndex}.localRT(trialIndex_removeOutlier));
        
        rt_std = std(performanceData{blockIndex}.localRT(trialIndex_removeOutlier));
        
        th_min = rt_mean - ksigma * rt_std;
        
        th_max = rt_mean + ksigma * rt_std;
        
         trialIndex = intersect( ...
             find(performanceData{blockIndex}.localRT > max(th_min,threshold_rt)), ...
             find(performanceData{blockIndex}.localRT < min(th_max,10)));
  
    else
        trialIndex = intersect( ...
            find(performanceData{blockIndex}.localRT > threshold_rt), ...
            find(performanceData{blockIndex}.localRT < 10));

    end
    

    
    %% 更新
    timestamp{sessionIndex} = performanceData{blockIndex}.deviatime(trialIndex);
    
    localRT{sessionIndex} = performanceData{blockIndex}.localRT(trialIndex);
    
    %% 计算方法1：指数函数归一化：
    tao_5th{sessionIndex} = prctile(localRT{sessionIndex},1);% 阈值选择：1%
   
    tao_0 = tao_5th{sessionIndex};

    % 1）90s平滑得global，对global归一化得nor_rt
    globalRT{sessionIndex} = movingwinForGlobalRT(localRT{sessionIndex},timestamp{sessionIndex},globalwin,'rt');
    
    globalDI_1{sessionIndex} = max(0, ...
        (1-exp(-(globalRT{sessionIndex}-tao_0)))...
        ./(1+exp(-(globalRT{sessionIndex}-tao_0)))); % local normalized to [0, 1]
    
    % 2）local归一化得localDI, 90s平滑得globalDI
    localDI{sessionIndex} = max(0, ...
        (1-exp(-(localRT{sessionIndex}-tao_0)))...
        ./(1+exp(-(localRT{sessionIndex}-tao_0))));
    
    globalDI_2{sessionIndex} = movingwinForGlobalRT(localDI{sessionIndex},timestamp{sessionIndex},globalwin,'rt');
    %% 增加变化范围
    globalDI_2{sessionIndex} = min(1,2*globalDI_2{sessionIndex});
    
    %% 计算方法2：线性归一化
    % 每个block前10个trial作为baseline
    RT10{sessionIndex} = mean(localRT{sessionIndex}(1:10));
    nor_localRT = (localRT{sessionIndex}-RT10{sessionIndex})/RT10{sessionIndex};
    nor_globalRT{sessionIndex} = movingwinForGlobalRT(nor_localRT,timestamp{sessionIndex},globalwin,'rt');
    
    
    try
    % eeg data 
    % 丢掉< 50ms trial
    if ~isempty(fea_alltrial)
        if strcmp(featureFilterMode, 'corranalysis')
            for chanIndex = 1:length(fea_alltrial{sessionIndex}{blockIndex}.allPSD) %chanNum
                temp = fea_alltrial{sessionIndex}{blockIndex}.allPSD{chanIndex};
                fea{sessionIndex}{chanIndex}.freq = temp.freq;
                fea{sessionIndex}{chanIndex}.srate = temp.srate;
                fea{sessionIndex}{chanIndex}.mixd = temp.mixd(:,trialIndex);
                if isfield(temp,'Beta')
                    fea{sessionIndex}{chanIndex}.frac = temp.frac(:,trialIndex);
                    fea{sessionIndex}{chanIndex}.osci = temp.osci(:,trialIndex);
                    fea{sessionIndex}{chanIndex}.beta = temp.Beta(trialIndex);
                    fea{sessionIndex}{chanIndex}.cons = temp.Cons(trialIndex);
                end
                
            end
        elseif strcmp(featureFilterMode, 'classification')
            temp = fea_alltrial{sessionIndex}{blockIndex}.featureMatrix(:,:,trialIndex); % chan * fea * trial
            % 平滑 脑电特征
%             temp = movingwinForGlobalRT(temp,timestamp{sessionIndex},globalwin,'eeg');
            
            fea{sessionIndex} = temp;
        end
        
    else
        fea = [];
    end
    catch 
        s=1;
             
    end
    
    %% 两类数据

    global_tao_5th{sessionIndex} =  prctile(globalDI_2{sessionIndex},20);% 阈值选择：10%
    global_tao_95th{sessionIndex} =  prctile(globalDI_2{sessionIndex},80);% 阈值选择：10%
    
    % 考虑时间（前10min）+RT
    trialNum = length(globalDI_2{sessionIndex});
     alertIndex{sessionIndex} = 1:min(60,trialNum/2); %
    drowsyIndex{sessionIndex} = intersect(find(globalDI_2{sessionIndex} >= global_tao_95th{sessionIndex}), ...
        min(61,trialNum/2+1):length(globalDI_2{sessionIndex}),'stable');    
    
    % 顺序打乱
    label_2class{sessionIndex} = [0 * ones(1,length(alertIndex{sessionIndex})), ...
        1 * ones(1,length(drowsyIndex{sessionIndex}))];
    
    trialIndex_2class = [alertIndex{sessionIndex},drowsyIndex{sessionIndex}];
    
    
    % 恢复顺序
    [trialIndex_2class, I] = sort(trialIndex_2class);
    label_2class{sessionIndex} = label_2class{sessionIndex}(I);
   
    
    globalDI_2_2class{sessionIndex} = globalDI_2{sessionIndex}(trialIndex_2class);
    
    localDI_2class{sessionIndex} = localDI{sessionIndex}(trialIndex_2class);
    
    nor_globalRT_2class{sessionIndex} = nor_globalRT{sessionIndex}(trialIndex_2class);
    
    globalRT_2class{sessionIndex} = globalRT{sessionIndex}(trialIndex_2class);
    
    localRT_2class{sessionIndex} = localRT{sessionIndex}(trialIndex_2class);
    
    tao_5th_2class{sessionIndex} = tao_5th{sessionIndex};
    
    RT10_2class{sessionIndex} = RT10{sessionIndex};

    timestamp_2class{sessionIndex} = timestamp{sessionIndex}(trialIndex_2class);
    
    % 所有trial的label
    label_alltrial{sessionIndex} = 2 * ones(1, trialNum);
    label_alltrial{sessionIndex}(alertIndex{sessionIndex}) = 0;
    label_alltrial{sessionIndex}(drowsyIndex{sessionIndex}) = 1;
    
    
    % eeg
    try
    if ~isempty(fea)
        if strcmp(featureFilterMode, 'corranalysis')  
        %% corranalysis
            for chanIndex = 1:chanNum

                temp = fea{sessionIndex}{chanIndex};
                fea_2class{sessionIndex}{chanIndex}.freq = temp.freq;
                fea_2class{sessionIndex}{chanIndex}.srate = temp.srate;
                fea_2class{sessionIndex}{chanIndex}.mixd = temp.mixd(:,trialIndex_2class);
                
                if isfield(temp,'beta')
                    fea_2class{sessionIndex}{chanIndex}.frac = temp.frac(:,trialIndex_2class);
                    fea_2class{sessionIndex}{chanIndex}.osci = temp.osci(:,trialIndex_2class);
                    fea_2class{sessionIndex}{chanIndex}.beta = temp.beta(trialIndex_2class);
                    fea_2class{sessionIndex}{chanIndex}.cons = temp.cons(trialIndex_2class);
                end
                
            end
        elseif strcmp(featureFilterMode, 'classification')
            %% classification
            temp = fea{sessionIndex}; % chan * fea * trial
            fea_2class{sessionIndex} = temp(:,:,trialIndex_2class);
        end
    else
        fea_2class = [];
    end
    catch 
        s=1;
    end
    
    end
    
end



%% 选择输出
    %% 输出所有trial，有连续rt（target），有类别标签（label）
    if strcmp(classficationMode,'alltrial') 
        
        % 特征
        fea_output = fea;
        timestamp_output = timestamp;
        
        
        % target & label
        if strcmp(targetName,'globalDI')
            target_output.target = globalDI_2;            
        elseif strcmp(targetName,'localDI')
            target_output.target = localDI;            
        elseif strcmp(targetName,'normalizedRT')
            target_output.target = nor_globalRT;            
        elseif strcmp(targetName,'globalRT')
            target_output.target = globalRT;            
        elseif strcmp(targetName,'localRT')
            target_output.target = localRT;            
        elseif strcmp(targetName,'tao_5th')
            target_output.target = tao_5th;
        elseif strcmp(targetName,'global_tao_5th')
            target_output.target = global_tao_5th;
        elseif strcmp(targetName,'global_tao_95th')
            target_output.target = global_tao_95th;
        elseif strcmp(targetName,'RT10')
            target_output.target = RT10;
        elseif strcmp(targetName,'KSSscore')
            target_output.target = kss_rt_score;
        else
            fprintf('请输入正确的反应时（target）返回类型-globalDI, normalizedRT, globalRT, localRT (默认globalDI) \n')
            target_output.target = globalDI_2;
        end
        
        target_output.label = label_alltrial; 
    
    %% 筛选两类数据，包含连续rt（target） 和 类别标签（label）
    elseif strcmp(classficationMode,'2class')
        % 特征
        fea_output = fea_2class;
        timestamp_output = timestamp_2class;
        
        
        % target & label
        if strcmp(targetName,'globalDI')
            target_output.target = globalDI_2_2class;
            
        elseif strcmp(targetName,'localDI')
            target_output.target = localDI_2class;
            
        elseif strcmp(targetName,'normalizedRT')
            target_output.target = nor_globalRT_2class;
            
        elseif strcmp(targetName,'globalRT')
            target_output.target = globalRT_2class;
            
        elseif strcmp(targetName,'localRT')
            target_output.target = localRT_2class;
            
        elseif strcmp(targetName,'tao_5th')
            target_output.target = tao_5th;
          
        elseif strcmp(targetName,'global_tao_5th')
            target_output.target = global_tao_5th;
            
        elseif strcmp(targetName,'global_tao_95th')
            target_output.target = global_tao_95th;
            
        elseif strcmp(targetName,'RT10')
            target_output.target = RT10;
            
        else
            printf('请输入正确的反应时（target）返回类型-globalDI, normalizedRT, globalRT, localRT (默认globalDI) \n')
            target_output.target = globalDI_2_2class;
        end
        
        target_output.label = label_2class; 
      
        
    end
        
end

