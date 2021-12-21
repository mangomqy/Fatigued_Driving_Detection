function [acc_all,C,prob,varargout] = classificationModel_v2(data,target,timestamp, classifier, methodMode,varargin)
%UNTITLED �˴���ʾ�йش˺�����ժҪ
%   output:
%   rmse: ƽ��ƽ����MSE (mean squared error)��
%   square_r: ƽ�����ϵ����r2 (squared correlation coefficient))

% 20210625


%% v2.0
if strcmp(methodMode,'crossSession')
    
    configData_v1;
    domainAdaptationMode = varargin{1};
    
    %% �����Դμ������� & ƽ������
    for sessionIndex = cat(2,subSet{:})
        %%  2class
        trialIndex = find(target.label{sessionIndex} ~= 2 );
        
        %% select channel
        eeg = data{sessionIndex}(:,:,:);
        
        %% psd feature: chan * dimension * trial
        calfea = feature_irasa(250);
        
        [feature_unsmooth{sessionIndex},~] = calfea.separating_main(eeg, frange, ...
            size(eeg,1), 'mixd', 'classification'); % chan * dimension * trial
        
        %% smooth egg
        
        feature_smooth{sessionIndex} = movingwinForGlobalRT(feature_unsmooth{sessionIndex}, timestamp{sessionIndex}, globalwin,'eeg');
        
        %% filter 2 class
        data_2class{sessionIndex} = feature_smooth{sessionIndex}(:,:,trialIndex); % chan * dimention * trial
        target_2class{sessionIndex} = target.label{sessionIndex}(trialIndex);
        
    end
    

    %% test
    for subIndex = subNumSet
        
        fprintf(['begin to process sub',num2str(subIndex),' \n'])
        
        sessionInput = subSet{subIndex};
        
        for j = 1:length(sessionInput)
            
            sessionIndex = sessionInput(j);
            
            testsession = sessionIndex;
            trainsession = setdiff(sessionInput,testsession);
            
            
            trainy = cat(2,target_2class{trainsession}); % source
            trainx = cat(3,data_2class{trainsession}); % chan * dimension * trial
            
            testy = cat(2,target_2class{testsession}); % target
            testx = cat(3,data_2class{testsession}); %
            


            
            %% ʹ��һ����target label����
            
            
            %% reshape
            trainy = trainy';
            trainx = (reshape(trainx,[],length(trainy)))';
            
            testy = testy';
            testx = (reshape(testx,[],length(testy)))';
            
            testy(1:8) = [];
            testx(1:8,:) = [];
            

            
            
            transy = testy(1:10);
            transx = testx(1:10,:);
            %% domain adaptation��testyֻ֪��ǰ10��
            if ~strcmp(domainAdaptationMode,'easyTL')
                [trainx, tex, transMdl] = domain_adaptation(trainx,trainy, transx,transy, domainAdaptationMode);
                testx = zscore(testx);
                testx = testx * transMdl.W;

               
                %% train model
                % default parameters
                
                if strcmp(classifier,'libsvm-linear')
                    model = svmtrain(trainy, trainx,'-s 0 -t 0 -c 1 ');
                    
                    % predict-ѵ����
                    [py,acc,~] = svmpredict(trainy, trainx, model,'-q');
                    [C{1,j},order] = confusionmat(trainy,py,'Order',[0,1]);
                    acc_all(j,1) = acc(1);
                    
                    % predict-���Լ�
                    [py,acc,prob{2,j}] = svmpredict(testy, testx, model);
                    [C{2,j},order] = confusionmat(testy,py,'Order',[0,1]);
                    acc_all(j,2) = acc(1);
                    
                end
                
            else
                
                [acc, py,y_prob] = domain_adaptation(trainx,trainy, testx,testy, domainAdaptationMode);
                
                acc_all(subIndex,j) = acc;
                
                [C{subIndex,j},order] = confusionmat(testy,py,'Order',[0,1]);
            end
            
        end
        
    end
    
    
end


end

%%
function show2classv2(trainx,trainy, testx,testy,trainsession,testsession,subIndex,varargin)
configData_v1;
methodMode = varargin{1};

if strcmp(methodMode, 'crossSession') || strcmp(methodMode, 'crossSubjectv2')
    % ѵ���������������ĵ�
    [source_class_center, ~]= get_class_center_yxy(trainx,trainy,testx,dist_trainx_center); % targetÿ����������source ���������ľ���
    
    
    featureMatrix_2class_forplot = cat(1,trainx,testx,source_class_center');
    label_2class = cat(1,trainy,testy);
    
    trainIndex = 1:length(trainy);
    testIndex = length(trainy)+1:length(label_2class);
    
    
    X_tsne = (tsne(featureMatrix_2class_forplot))';
    
    
    
    %%
    % train
    gscatter(X_tsne(1,trainIndex)', X_tsne(2,trainIndex), num2str(label_2class(trainIndex)),'rk','.'),hold on
    % test
    gscatter(X_tsne(1,testIndex)', X_tsne(2,testIndex), num2str(label_2class(testIndex)),'rk','x'),hold on
    
    %     ѵ�������ĵ�
    gscatter(X_tsne(1,end-1)', X_tsne(2,end-1), '0','c','.',12,'off'),hold on
     gscatter(X_tsne(1,end)', X_tsne(2,end), '1','b','.',12,'off'),hold on
    
    legend('off')
    
    title(['sub ',num2str(subIndex),': session ',num2str(trainsession),...
        ' transfer to session ',num2str(testsession)])
    hold off
    
end
end
