classdef corranalysis
    %UNTITLED �˴���ʾ�йش����ժҪ
    %   �˴���ʾ��ϸ˵��
    
    properties
        % eeg
        fea_alltrial
        fea
        
        % performance
        performanceData_rt_alltrial % allevent & all trial
        target
        label
        performanceData_kss
        preprocessedParams
        performanceData_alltrigger
        
        % with time
        timestamp
        chanlocs
    end
    
    methods
        %% initial
        function obj = corranalysis(project)
            % corranalysis ��������ʵ��
            configData_v1;
                       
            
            %% load performance data
            obj = obj.computePerformanceRT_v1(project);
            
        end
        
        
        function obj = computePerformanceRT_v1(obj, project)
            configData_v1;
            %% performance data
                    
            sessionIndex = 0;
            
            for EEGFileIndex = 1:length(project.processedList) 
                
                
                subName = project.processedList{EEGFileIndex}(1:3);
                
                fileName = project.processedList{EEGFileIndex}(5:end);
                
                blockIndex = str2num(project.processedList{EEGFileIndex}(end));
                
                if blockIndex == 1
                    sessionIndex = sessionIndex + 1;
                end
                
                fprintf(['processing session ',num2str(sessionIndex),'...\n']);
                fprintf(['processing session ',fileName,'\n']);
                
                finalPerformName = [EEGperformDataPath,'/',subName,'/',fileName,'.mat'];
                if ~exist(finalPerformName,'file')
                    fprintf('��Ϊѧ���ݲ����ڣ����ڼ���... \n')
                    %% load preprocessed eeg data
                    finalDataFilePath = [resultsFolder,'/',subName,'/nip_',fileName,'.mat'];
                    
                    finalData = load(finalDataFilePath);
                    
                    EEG = finalData.EEG;
                    
                    %% rt�����������trigger�ֲ�
                    performanceData = TESTcalPerformance(EEG.event,EEG.srate,'filteredEvent');
                    performanceData_rt= performanceData;
                    
                    
                    %% �������
                    
                    performanceData_alltrigger = EEG.event;
                    if ~isfolder([EEGperformDataPath,'/',subName])
                        mkdir(EEGperformDataPath,subName)
                    end
                    save(finalPerformName, 'performanceData_rt','performanceData_alltrigger')
                else
                    fprintf('��Ϊѧ���ݴ��ڣ����ڼ���... \n')
                    load(finalPerformName)
                end
                performanceData_rt_all{sessionIndex}{blockIndex} = performanceData_rt;
                performanceData_alltrigger_all{sessionIndex}{blockIndex} = performanceData_alltrigger;
                
            end
            
                       
            obj.performanceData_rt_alltrial = performanceData_rt_all;
            
            obj.performanceData_alltrigger = performanceData_alltrigger_all;
            
        end

        function corranalysis_plot_spectrum_rt(obj, componentMode, MethodMode, subInput, targetName)
            calfea = feature_irasa(250);
            
            [feature_unsmooth{sessionIndex},~] = calfea.separating_main(eeg, frange, ...
                size(eeg,1), 'mixd', 'classification'); % chan * dimension * trial
            plot_spectrum_rt(obj.fea, obj.target, componentMode, MethodMode, subInput, targetName)
        end
        

    end
end