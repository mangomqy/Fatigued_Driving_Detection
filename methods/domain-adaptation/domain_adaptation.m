function [trainx_out, testx_out, transMdl] = domain_adaptation(trainx,trainy, testx,testy,methodMode,varargin)

% input
% trainx: trial * feature




trainTrialNum = length(trainy);
testTrialNum = length(testy);

% 关于核函数：
% 如果特征数量很大，跟样本数量差不多，往往数据在原空间的可分性就比较大。因此，可选择线性核，极大较少计算量。
% 如果特征数量少，而样本数量不大。考虑使用非线性核如高斯核，将数据映射到无穷维可分的空间。


%% 共同子空间
% TCA：完全无监督
if strcmp(methodMode, 'tca')
    % 重要参数： kerSigma 正则项mu，降维特征数 m
    param = []; param.kerName = 'lin';param.bSstca = 0;
    param.mu = 10;param.gamma = 1;param.lambda = 1;%0
    param.m = size(trainx,1); % 特征数
    param.m = 30; % 特征数
%     param.m = varargin{1};
    
    
    X = cat(1,trainx, testx);
    maSrc = logical(cat(1, 1*ones(trainTrialNum,1), 0*ones(testTrialNum,1))); % 1 source domain
    % label: only source(实际)
    target = cat(1,trainy);
    maLabeled = logical(cat(1, 1*ones(trainTrialNum,1), 0*ones(testTrialNum,1))); % 是否标记
    
    [Xproj,transMdl] = ftTrans_tca(X,maSrc,target,maLabeled,param); % 输出特征维数不能超过样本数？
    
    
    trainx_out = Xproj(1:trainTrialNum, :);
    testx_out =  Xproj(trainTrialNum+1:end, :);

elseif strcmp(methodMode, 'tca_v2') 
    X_src = trainx;
    X_tar = testx;
    
    options.lambda = 1;
    options.dim = 30;
    options.kernel_type = 'primal'; % 'primal' | 'linear' | 'rbf' kernel影响大
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
    % 重要参数： kerSigma 正则项mu，降维特征数 m
    param = []; param.kerName = 'lin';param.bSstca = 1;
    param.mu = 1;param.gamma = .1;param.lambda = 0;
    param.m = size(trainx,1); % 特征数
    param.m = 60; % 特征数
    
    X = cat(1,trainx, testx);
    maSrc = logical(cat(1, 1*ones(trainTrialNum,1), 0*ones(testTrialNum,1))); % 1 source domain
    % label: only source
    target = cat(1,trainy);
    maLabeled = logical(cat(1, 1*ones(trainTrialNum,1), 0*ones(testTrialNum,1))); % 是否标记
    % label: both source + target
%      target = cat(1,trainy,testy);
%      maLabeled = logical(cat(1, 1*ones(trainTrialNum,1), 1*ones(testTrialNum,1))); % 是否标记
    
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
    maLabeled = logical(cat(1, 1*ones(trainTrialNum,1), 0*ones(testTrialNum,1))); % 是否标记
    % label: both source + target
%      target = cat(1,trainy,testy);
%      maLabeled = logical(cat(1, 1*ones(trainTrialNum,1), 1*ones(testTrialNum,1))); % 是否标记
    
    [Xproj,transMdl] = ftTrans_mida(X,maSrc,target,maLabeled,param);
    
    trainx_out = Xproj(1:trainTrialNum, :);
testx_out =  Xproj(trainTrialNum+1:end, :);
   
    
    % sa
elseif strcmp(methodMode, 'sa')
    param = []; param.pcaCoef = 2;
    
    X = cat(1,trainx, testx);
    maSrc = logical(cat(1, 1*ones(trainTrialNum,1), 0*ones(testTrialNum,1))); % 1 source domain
    target = cat(1,trainy,testy);
    maLabeled = logical(cat(1, 1*ones(trainTrialNum,1), 1*ones(testTrialNum,1))); % 是否标记
    
    
    [Xproj,transMdl] = ftTrans_sa(X,maSrc,target,maLabeled,param);
    
    trainx_out = Xproj(1:trainTrialNum, :);
testx_out =  Xproj(trainTrialNum+1:end, :);
   



    % itl
elseif strcmp(methodMode, 'itl') % 无监督
    
    param = []; param.pcaCoef = 1; param.lambda = 10;
    
    
    X = cat(1,trainx, testx);
    maSrc = logical(cat(1, 1*ones(trainTrialNum,1), 0*ones(testTrialNum,1))); % 1 source domain
    target = trainy;
    maLabeled = maSrc; % 是否标记
    
    
    [Xproj,transMdl] = ftTrans_itl(X,maSrc,target,maLabeled,param);
    
    trainx_out = Xproj(1:trainTrialNum, :);
testx_out =  Xproj(trainTrialNum+1:end, :);
  
%% 流形空间
% gfk:无监督
elseif strcmp(methodMode, 'gfk') 
    % error:用于矩阵乘法的维度不正确。  pca版本太多导致报错
    
    param = []; 
    param.dr = 1; % ratio that controls the dimension of the subspace.If 0, will be 
		% automatically computed according to ref 1.
    param.bSup = 0;
%     fprintf(['')
    
    X = cat(1,trainx, testx);
    maSrc = logical(cat(1, 1*ones(trainTrialNum,1), 0*ones(testTrialNum,1))); % 1 source domain
    target = trainy;
    maLabeled = maSrc; % 是否标记
    
    % 输出特征维度:
    % dr = 0: 24*2
    % dr = 1:输入特征的一半（并取整） 72
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
    
    % 计算投影矩阵
    [TL,TD]=ldl(G);
    L=TL*(TD.^0.5);
    % 	A = chol(G+eps*20*eye(size(G,1)));
    % 	L = A'; % similar to ldl
    transMdl.W = real(L(:,1:dim)); % imaginary after d*2 because rank deficient
    
    % 数据投影
    trainx_out = X_src * transMdl.W;
     testx_out = X_tar * transMdl.W;
  
     
     
 % easyTL
elseif strcmp(methodMode, 'easyTL') % 
    
    Xs = trainx; % trial * dimension
    Ys = trainy;
    Xt = testx;
    Yt = testy;
    
    
    
    [acc,y_pred,y_prob] = EasyTL(Xs,Ys,Xt,Yt,'coral'); % 默认coral
    
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
   

    
    
 %% 映射到target所在子空间
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
% maLabeled = logical(cat(1, 1*ones(trainTrialNum,1), 1*ones(testTrialNum,1))); % 是否标记
%
% [Xproj,transMdl] = ftTrans_tca(X,maSrc,target,maLabeled,param); % 输出特征维数不能超过样本数？

%%



end

