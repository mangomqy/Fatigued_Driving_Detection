classdef classification < corranalysis
    % 继承自corranalysis.m
    properties
        project
        
        epochEEG_2class
        y_2class
        timestamp_2class
        
        fbEpochEEG_2class
        
        acc
        C
        order
        
        rejectRatio
        
        trialnum
        channum
        
        session_similarity
        sourceSession
        similarity
    end
    
    methods
        %% initial
        function obj = classification(project)
            
            % classification 构造此类的实例
            obj = obj@corranalysis(project);
            obj.epochEEG_2class = [];
            obj.rejectRatio = [];
            configData_v1;
%             chanlocs = readlocs(locPath63);
%             obj.chanlocs = chanlocs;
            
            obj.project = project;
        end
        
        
        function obj = classification_filter_trial_v2(obj,snorkelMode,trialSelectMode,classficationMode,featureFilterMode,targetName)
            
            if strcmp(snorkelMode, 'nosnorkel')
                [~,y,timestamp] = filter_trial_v2([],obj.performanceData_rt_alltrial,...
                    trialSelectMode,classficationMode,featureFilterMode,targetName);
                obj.y_2class = y;
                obj.timestamp_2class = timestamp;
            end
            
        end
        
        
        function obj = classification_epoch(obj, project,chanSelectMode,varargin)
            
            configData_v1
            %% get epoch data
            clear epochEEG_2class
            
            sessionIndex = 0;
            
            keySet = {'frontal','temporal','central','parietal','occipital','behindEar'};
            valueSet = {[1:5,8:10], [11:16,20:25,31:34,39:42,48:49], [17:19,26:30],...
                [35:38,43:47], [50:52,55:59], [6,7,53,54,61:64]};
            
            chanSubPositionDict = containers.Map(keySet,valueSet);
            
            if strcmp(chanSelectMode,'allChannel')
                chanSelect = 1:63;
            elseif strcmp(chanSelectMode,'rejectOutlier')
                chanSelect = setdiff(1:63, ...
                    [1:3,6:7,15:16,24:25,33:34,41:42,48:49,55:63]);
            elseif strcmp(chanSelectMode,'behindEar')
                chanSelect = [6,7,53,54,61:64];
            else
                chanSelect = chanSubPositionDict(chanSelectMode);
            end
            
            
            for EEGFileIndex = 1:length(cat(2,subSet{:})) % project.processedList: all session & block
                
                
                subName = project.processedList{EEGFileIndex}(1:3); % s07
                %     subIndex = str2num(project.processedList{sessionIndex}(2:3));
                
                fileName = project.processedList{EEGFileIndex}(5:end);
                
                blockIndex = str2num(project.processedList{EEGFileIndex}(end));
                
                if blockIndex == 1
                    sessionIndex = sessionIndex + 1;
                end
                
                fprintf(['processing session ',num2str(sessionIndex),'...\n']);
                fprintf(['processing session ',fileName,'\n']);
                
                %% load preprocessed eeg data
                finalDataFilePath = [resultsFolder,'/',subName,'/nip_',fileName,'.mat'];
                finalEpochFilePath = [EEGepochDataPath,'/',subName,'/',fileName,'.mat'];
                if ~exist(finalEpochFilePath,'file')
                    
                finalData = load(finalDataFilePath);
                
                EEG = finalData.EEG;
                
                EEG.data = double(EEG.data(chanSelect,:)); % 64 is ref
                EEG.nbchan = length(chanSelect);
                EEG.chanlocs = EEG.chanlocs(chanSelect);
                
                %% bandpass
                if length(varargin)>1
                    nFB = varargin{2};
                    try
                        
                        EEG.data = bpcsp_filter(EEG.data,250,nFB);
                        %                         y = eegfiltfft(x,srate,[],13);
                    catch
                        fprintf('fbcsp缺少nFB输入\n')
                    end

                    
                end
                %% eeg epoch
                % all events
                %     events = performanceData_rt{sessionIndex}{blockIndex}.deviatime;
                % 2class events
                events = obj.timestamp_2class{sessionIndex};
                EEG.data_epoch = epoch(EEG.data, events, [-5,0], 'srate', EEG.srate);
                
                obj.epochEEG_2class{sessionIndex} =  EEG.data_epoch; % channel * timepoint * trial
                end
                
            end
            
        end
        
        % 根据eeg筛选trial: 需要epoch之后
        function obj = classification_filter_trial_eeg(obj)
            
            configData_v1;
            
            %
            if isempty(obj.epochEEG_2class)
                fprintf('please epoch data')
                
            else
                
                
                for sessionIndex = cat(2,subSet{:})
                    
                    signal = obj.epochEEG_2class{sessionIndex};
                    
                    frames = size(signal, 2);
                    negthresh = -100;
                    posthresh = 100;
                    timerange = [0,5]; %s?
                    startime = 0;
                    endtime = 5;
                    
                    
                    
                    % reject extreme values
                    [Iin{sessionIndex}, Iout{sessionIndex}, newsignal{sessionIndex}, elec{sessionIndex}] = ...
                        eegthresh(signal, frames, 1:8, ...
                        negthresh,posthresh, timerange, startime, endtime);
                    
                    % 丢弃比例
                    rejectRatio(sessionIndex) = length(Iout{sessionIndex})/size(signal,3);
                    
                    label{sessionIndex} = obj.y_2class.label{sessionIndex}(Iin{sessionIndex});
                    target{sessionIndex} = obj.y_2class.target{sessionIndex}(Iin{sessionIndex});
                    
                    timestamp{sessionIndex} = obj.timestamp_2class{sessionIndex}(Iin{sessionIndex});
                    
                    
                    
                    
                    
                end
                obj.y_2class.label = label;
                obj.y_2class.target = target;
                obj.timestamp_2class = timestamp;
                
                obj.epochEEG_2class = newsignal;
                
                obj.rejectRatio = rejectRatio;
                
            end
            
        end
        
        %检查数据质量
        function classification_checkSingleEpoch(obj,project, sessionIndex)
            configData_v1;
            %              figure()
            eegplot(obj.epochEEG_2class{sessionIndex},...%(1:63,:,:)
                'srate', project.sRate/project.dsRate, 'eloc_file',locPath63,...
                'winlength',3,...
                'title',['preprocessedData of session ', num2str(sessionIndex)])
            
            
        end
        
               
        function obj= classification_classificationModel_v2(obj, expNum, classifier, methodMode,varargin)
            
            obj.acc = [];
            configData_v1;
            
            
            if strcmp(methodMode,'crossSession') || strcmp(methodMode,'crossSubject')
                %% 切所有trial数据
                for i = 1:expNum
                    fprintf([num2str(i), ' th exp ... \n'])
                    [acc_all, C, probability] = classificationModel_v2(obj.epochEEG_2class, obj.y_2class,...
                        obj.timestamp_2class, classifier, methodMode,varargin{1}); % varagin
                    obj.acc(:,:,i) = acc_all;
                end
                obj.acc = mean(obj.acc,3);
                obj.C = C;
                obj.order = probability;
            elseif strcmp(methodMode,'crossSubjectv2')
                
                obj.session_similarity = [];
                obj.sourceSession = [];
                
                for i = 1:expNum
                    fprintf([num2str(i), ' th exp ... \n'])
                    [acc_all, C, order, session_similarity,sourceSession] = classificationModel_v2(obj.epochEEG_2class, obj.y_2class,...
                        obj.timestamp_2class, classifier, methodMode,varargin{1},varargin{2}); % varagin
                    obj.acc(:,:,i) = acc_all;
                end
                obj.acc = mean(obj.acc,3);
                obj.C = C;
                obj.order = order;
                obj.session_similarity = session_similarity;
                obj.sourceSession = sourceSession;
                
                %% cp_decopose
            elseif strcmp(methodMode,'cp_decompose')
                
                obj.session_similarity = [];
                obj.sourceSession = [];
                [test_similarity] = cal_similarity(obj.epochEEG_2class, obj.y_2class,...
                        obj.timestamp_2class,varargin{1}); 
                obj.similarity = test_similarity;
                
                %%   withinSession
            elseif strcmp(methodMode,'withinSession')
                
                cspMode = varargin{1};
                
                
                if strcmp(cspMode, 'csp')
                    
                    for i = 1:expNum
                        % classification
                        [acc_all, C, order] = classificationModel_v2(obj.epochEEG_2class, obj.y_2class,...
                            obj.timestamp_2class, classifier, methodMode); % varagin
                        obj.acc(:,:,i) = acc_all;
                    end
                    obj.acc = mean(obj.acc,3);
                    obj.C = C;
                    obj.order = order;
                    
                elseif strcmp(cspMode, 'fbcsp')
                    
                    % filter trial：2class
                    trialSelectMode = 'ksigma';  % 对rt筛选
                    targetName = 'globalDI';
                    % 根据rt筛trial 2class
                    obj = obj.classification_filter_trial_v2('nosnorkel',...
                        trialSelectMode, '2class', 'classification', targetName);
                    %
                    
                    for fbIndex  = 1:4
                        % epoch data
                        obj = obj.classification_epoch(obj.project,'allChannel','fbcsp',4);% allChannel
                        
                        % 根据eeg二次筛trial
                        obj = obj.classification_filter_trial_eeg;
                        fprintf('reject bad trial ratio (percent) : \n'); nonzeros(obj.rejectRatio * 100)
                        
                        %
                        temp_2class{fbIndex} = obj.epochEEG_2class;
                        
                        
                    end
                    
                    chancount = 0;
                    for fbIndex  = 1:4
                        chanindex = [chancount+1:chancount+obj.channum];
                        chancount = chancount+obj.channum;
                        for sessionIndex = cat(2,subSet{:})
                            fbEpochEEG_2class{sessionIndex}(chanindex, :,:) = temp_2class{fbIndex}{sessionIndex}; % chan * time * trial
                        end
                    end
                    
                    obj.fbEpochEEG_2class = fbEpochEEG_2class;
                    
                    % classification
                    [acc_all, C, order] = classificationModel_v2(obj.fbEpochEEG_2class, obj.y_2class,...
                        obj.timestamp_2class, classifier, methodMode); % varagin
                    
                    
                end
            end
            
            
            
            
            
        end
        

        function classfication_confusion_matrix(obj)
            configData;
            figure(),
            count = 0;
            
            
            for subIndex = subNumSet
                sessionInput = subSet{subIndex};
                for i = 1:length(sessionInput)
                    count = count+1;
                    subplot(6,6,count)
                    %     imagesc(C{subIndex,i})
                    confusionchart(obj.C{subIndex,i},obj.order);
                    title(['sub ',num2str(subIndex), ...
                        ' session ',num2str(sessionInput(i))])
                end
                
            end
        end
        

        function obj = plot_frequency_spectrum(obj,project)
            
            % load data
            configData_v1;
            sessionIndex = 0;
            for EEGFileIndex = 1:length(cat(2,subSet{:})) % project.processedList: all session & block
                
                
                subName = project.processedList{EEGFileIndex}(1:3); % s07
                %     subIndex = str2num(project.processedList{sessionIndex}(2:3));
                
                fileName = project.processedList{EEGFileIndex}(5:end);
                
                blockIndex = str2num(project.processedList{EEGFileIndex}(end));
                
                if blockIndex == 1
                    sessionIndex = sessionIndex + 1;
                end
                
                fprintf(['processing session ',num2str(sessionIndex),'...\n']);
                fprintf(['processing session ',fileName,'\n']);
                
                finalDataFilePath = [resultsFolder,'/',subName,'/nip_',fileName,'.mat'];
                finalData = load(finalDataFilePath);
                
                EEG = finalData.EEG;
                chanSelect = [6,7,53,54,61:64];
                EEG.data = double(EEG.data(chanSelect,:)); % 64 is ref
            end
            raw_data = EEG.data;
            time_second = length(raw_data)/250;
            eeg = zeros(8,1250,time_second);
            for trial_temp = 1:time_second/5
                eeg(:,:,trial_temp) = raw_data(:,(trial_temp-1)*1250+1:trial_temp*1250);
            end
            % calculate_ spectrum
            calfea = feature_irasa(250);
            frange = [1,30];
            [feature_unsmooth{sessionIndex},~] = calfea.separating_main(eeg, frange, ...
                size(eeg,1), 'mixd', 'both'); % chan * dimension * trial
            feature_all = zeros(59,time_second,8);
            for channel_index = 1:8
                feature_all(:,:,channel_index) = feature_unsmooth{1,1}.allPSD{1,channel_index}.mixd;
            end
            freq = feature_unsmooth{1,1}.allPSD{1,1}.freq(5:end);
                % plot_spectrum
            
            for trial_temp = 1:time_second/5
                pause(5)
                x = [1,8];
                y = [3, max(freq)];
                temp_freq =  log10(squeeze(feature_all(:,trial_temp,:)));
%                 imagesc(x,y,temp_freq(5:end,:));
                imagesc(y,x,temp_freq(5:end,:)');
                ylabel('channel')
                xlabel('freq (Hz)')
                c = colorbar;
                c.Label.String = 'power';
                colormap('jet')
                title('spectrum')
            end
        end
    end
end

