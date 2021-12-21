function [trainx_out, testx_out, transMdl] = domain_adaptation(trainx,trainy, testx,testy,methodMode,varargin)

% input
% trainx: trial * feature




trainTrialNum = length(trainy);
testTrialNum = length(testy);

% ���ں˺�����
% ������������ܴ󣬸�����������࣬����������ԭ�ռ�Ŀɷ��ԾͱȽϴ���ˣ���ѡ�����Ժˣ�������ټ�������
% ������������٣��������������󡣿���ʹ�÷����Ժ����˹�ˣ�������ӳ�䵽����ά�ɷֵĿռ䡣


%% ��ͬ�ӿռ�
% TCA����ȫ�޼ල
if strcmp(methodMode, 'tca')
    % ��Ҫ������ kerSigma ������mu����ά������ m
    param = []; param.kerName = 'lin';param.bSstca = 0;
    param.mu = 10;param.gamma = 1;param.lambda = 1;%0
    param.m = size(trainx,1); % ������
    param.m = 30; % ������
%     param.m = varargin{1};
    
    
    X = cat(1,trainx, testx);
    maSrc = logical(cat(1, 1*ones(trainTrialNum,1), 0*ones(testTrialNum,1))); % 1 source domain
    % label: only source(ʵ��)
    target = cat(1,trainy);
    maLabeled = logical(cat(1, 1*ones(trainTrialNum,1), 0*ones(testTrialNum,1))); % �Ƿ���
    
    [Xproj,transMdl] = ftTrans_tca(X,maSrc,target,maLabeled,param); % �������ά�����ܳ�����������
    
    
    trainx_out = Xproj(1:trainTrialNum, :);
    testx_out =  Xproj(trainTrialNum+1:end, :);

elseif strcmp(methodMode, 'tca_v2') 
    X_src = trainx;
    X_tar = testx;
    
    options.lambda = 1;
    options.dim = 30;
    options.kernel_type = 'primal'; % 'primal' | 'linear' | 'rbf' kernelӰ���
    options.gamma = 1; % 
    
    [X_src_new,X_tar_new,A] = TCA(X_src,X_tar,options);
    
    trainx_out = X_src_new;
    testx_out = X_tar_new;
    transMdl = A;
    
  
    
    
    % bda 
  elseif strcmp(methodMode, 'bda')
    
     Xs = trainx; % trial * dimension
%     Ys = trainy;
    Xt = testx;
%     Yt = testy;

    [trainx_out] = BDA(Xs,Xt);
    testx_out = Xt;
    
     transMdl = [];
    
    
% SSTCA
elseif strcmp(methodMode, 'sstca')
    % ��Ҫ������ kerSigma ������mu����ά������ m
    param = []; param.kerName = 'lin';param.bSstca = 1;
    param.mu = 1;param.gamma = .1;param.lambda = 0;
    param.m = size(trainx,1); % ������
    param.m = 60; % ������
    
    X = cat(1,trainx, testx);
    maSrc = logical(cat(1, 1*ones(trainTrialNum,1), 0*ones(testTrialNum,1))); % 1 source domain
    % label: only source
    target = cat(1,trainy);
    maLabeled = logical(cat(1, 1*ones(trainTrialNum,1), 0*ones(testTrialNum,1))); % �Ƿ���
    % label: both source + target
%      target = cat(1,trainy,testy);
%      maLabeled = logical(cat(1, 1*ones(trainTrialNum,1), 1*ones(testTrialNum,1))); % �Ƿ���
    
    [Xproj,transMdl] = ftTrans_tca(X,maSrc,target,maLabeled,param);
    
    trainx_out = Xproj(1:trainTrialNum, :);
testx_out =  Xproj(trainTrialNum+1:end, :);
 



    % MIDA 
elseif strcmp(methodMode, 'mida')
    
        param = []; param.kerName = 'lin';param.kerSigma = 1e-1;param.bSup = 0;
    param.mu = 1;param.gamma = 1;

    param.m = 30;
    
    X = cat(1,trainx, testx);
    maSrc = logical(cat(1, 1*ones(trainTrialNum,1), 0*ones(testTrialNum,1))); % 1 source domain
    % label: only source
    target = cat(1,trainy);
    maLabeled = logical(cat(1, 1*ones(trainTrialNum,1), 0*ones(testTrialNum,1))); % �Ƿ���
    % label: both source + target
%      target = cat(1,trainy,testy);
%      maLabeled = logical(cat(1, 1*ones(trainTrialNum,1), 1*ones(testTrialNum,1))); % �Ƿ���
    
    [Xproj,transMdl] = ftTrans_mida(X,maSrc,target,maLabeled,param);
    
    trainx_out = Xproj(1:trainTrialNum, :);
testx_out =  Xproj(trainTrialNum+1:end, :);
   
    
    % sa
elseif strcmp(methodMode, 'sa')
    param = []; param.pcaCoef = 2;
    
    X = cat(1,trainx, testx);
    maSrc = logical(cat(1, 1*ones(trainTrialNum,1), 0*ones(testTrialNum,1))); % 1 source domain
    target = cat(1,trainy,testy);
    maLabeled = logical(cat(1, 1*ones(trainTrialNum,1), 1*ones(testTrialNum,1))); % �Ƿ���
    
    
    [Xproj,transMdl] = ftTrans_sa(X,maSrc,target,maLabeled,param);
    
    trainx_out = Xproj(1:trainTrialNum, :);
testx_out =  Xproj(trainTrialNum+1:end, :);
   



    % itl
elseif strcmp(methodMode, 'itl') % �޼ල
    
    param = []; param.pcaCoef = 1; param.lambda = 10;
    
    
    X = cat(1,trainx, testx);
    maSrc = logical(cat(1, 1*ones(trainTrialNum,1), 0*ones(testTrialNum,1))); % 1 source domain
    target = trainy;
    maLabeled = maSrc; % �Ƿ���
    
    
    [Xproj,transMdl] = ftTrans_itl(X,maSrc,target,maLabeled,param);
    
    trainx_out = Xproj(1:trainTrialNum, :);
testx_out =  Xproj(trainTrialNum+1:end, :);
  
%% ���οռ�
% gfk:�޼ල
elseif strcmp(methodMode, 'gfk') 
    % error:���ھ���˷���ά�Ȳ���ȷ��  pca�汾̫�ർ�±���
    
    param = []; 
    param.dr = 1; % ratio that controls the dimension of the subspace.If 0, will be 
		% automatically computed according to ref 1.
    param.bSup = 0;
%     fprintf(['')
    
    X = cat(1,trainx, testx);
    maSrc = logical(cat(1, 1*ones(trainTrialNum,1), 0*ones(testTrialNum,1))); % 1 source domain
    target = trainy;
    maLabeled = maSrc; % �Ƿ���
    
    % �������ά��:
    % dr = 0: 24*2
    % dr = 1:����������һ�루��ȡ���� 72
    [Xproj,transMdl] = ftTrans_gfk(X,maSrc,target,maLabeled,param); 
    
    trainx_out = Xproj(1:trainTrialNum, :);
    
    testx_out =  Xproj(trainTrialNum+1:end, :);
    
elseif strcmp(methodMode, 'gfk_v2') % 
    
     X_src = trainx; % trial * dimension
     Y_src = trainy;
    X_tar = testx;
    Y_tar = testy;
    dim = 30;
    
    % GFL + 1NN
    [acc,G,Cls] = GFK_v2(X_src,Y_src,X_tar,Y_tar,dim);
    
    fprintf(['acc of 1NN classifier = ', num2str(acc),'\n'])
    
    % ����ͶӰ����
    [TL,TD]=ldl(G);
    L=TL*(TD.^0.5);
    % 	A = chol(G+eps*20*eye(size(G,1)));
    % 	L = A'; % similar to ldl
    transMdl.W = real(L(:,1:dim)); % imaginary after d*2 because rank deficient
    
    % ����ͶӰ
    trainx_out = X_src * transMdl.W;
     testx_out = X_tar * transMdl.W;
  
     
     
 % easyTL
elseif strcmp(methodMode, 'easyTL') % 
    
    Xs = trainx; % trial * dimension
    Ys = trainy;
    Xt = testx;
    Yt = testy;
    
    
    
    [acc,y_pred,y_prob] = EasyTL(Xs,Ys,Xt,Yt,'coral'); % Ĭ��coral
    
    y_pred = y_pred - 1;
    
    fprintf(['acc of easyTL = ', num2str(acc),'\n'])
    
    
    
    trainx_out = acc;
    testx_out = y_pred;
    transMdl = y_prob;

elseif strcmp(methodMode, 'easyTL_yxy') %
    
    Xs = trainx; % trial * dimension
    Ys = trainy;
    Xt = testx;
    Yt = testy;
    
    
    
    [trainx_out, testx_out] = EasyTL_yxy(Xs,Ys,Xt,Yt,'gfk');
    
    transMdl = [];
   

    
    
 %% ӳ�䵽target�����ӿռ�
    % CORAL: 
elseif strcmp(methodMode, 'coral')
    
     Xs = trainx; % trial * dimension
%     Ys = trainy;
    Xt = testx;
%     Yt = testy;

    [trainx_out] = CORAL(Xs,Xt);
    testx_out = Xt;
    
     transMdl = [];
    
    
    
    
elseif strcmp(methodMode, 'none') %
    
    trainx_out = trainx;
    
    testx_out = testx;
    
    transMdl = [];
    
end




%% 1.0
% train_eeg: chan * time point * trial
% trainy: 1:trial
%


% trainTrialNum = length(trainy);
% testTrialNum = length(testy);
%
% train_eeg_reshape = reshape(train_eeg,[],trainTrialNum);
% train_eeg_reshape = train_eeg_reshape'; % trial * feature
% trainy_reshape = trainy';
%
%
% test_eeg_reshape = reshape(test_eeg,[],testTrialNum);
% test_eeg_reshape = test_eeg_reshape'; % trial * feature
% testy_reshape = testy';
%
% X = cat(1,train_eeg_reshape, test_eeg_reshape);
% maSrc = logical(cat(1, 1*ones(trainTrialNum,1), 0*ones(testTrialNum,1))); % 1 source domain
% target = cat(1,trainy_reshape,testy_reshape);
% maLabeled = logical(cat(1, 1*ones(trainTrialNum,1), 1*ones(testTrialNum,1))); % �Ƿ���
%
% [Xproj,transMdl] = ftTrans_tca(X,maSrc,target,maLabeled,param); % �������ά�����ܳ�����������

%%



end

