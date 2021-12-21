function plot_spectrum_rt(fea,normalizedrt,componentMode, MethodMode,subInput,targetName)
% varargin:
% 1: 'mixd', 'frac','osci'
% 2: 'meanSub': 所有的输入受试一起平均
%    'singleSubAllSession'：对于输入的【单个】受试，分别展示每个session
% 3: 如果2为'meanSub'或'singleSubAllSession'，则可以选择输入sub

configData

freq = fea{end}{1}.freq;

for sessionIndex = sessionSet
    for chanIndex = 1:chanNum
        if strcmp(componentMode,'mixd')
            eegspectrum{sessionIndex}(chanIndex,:,:) = fea{sessionIndex}{chanIndex}.mixd;% chan, freq, trial
            
        elseif strcmp(componentMode,'frac')
            eegspectrum{sessionIndex}(chanIndex,:,:) = fea{sessionIndex}{chanIndex}.frac;% chan, freq, trial
            
        elseif strcmp(componentMode,'osci')
            eegspectrum{sessionIndex}(chanIndex,:,:) = fea{sessionIndex}{chanIndex}.osci;% chan, freq, trial
        end
        
        
    end
end



if strcmp(targetName,'localRT') || strcmp(targetName,'globalRT')
    rt_normalized_range = 0.1:0.01:2.5;
    rt_normalized_step = 0.1;
    x = [0,2.5];
else
    rt_normalized_range = 0.01:0.001:0.99;
    rt_normalized_step = 0.01;
    x = [0,1];
end

figure()

%cat(3,temp{:})-> channel ,freq, trial

if strcmp(MethodMode,'meanSubOnePlot')
    
    %     subInput = varargin{3};
    
    eegspectrum_reshape = max(0,cat(3,eegspectrum{cat(2,subSet{subInput})}));
    
    [normalizedrt_sort,sortIndex] = sort(cat(2, normalizedrt{cat(2,subSet{subInput})}));
    
    if strcmp(componentMode,'mixd') || strcmp(componentMode,'frac')
        
        eegspectrumList_sort = log10(squeeze(mean(eegspectrum_reshape(:,:,sortIndex),1)));
        
    elseif strcmp(componentMode,'osci')
        
        eegspectrumList_sort = squeeze(mean(eegspectrum_reshape(:,:,sortIndex),1));
        
    end
    
    
    
    
    for i = 1:length(rt_normalized_range)
        temp(:,i) = mean(eegspectrumList_sort(:,...
            find(normalizedrt_sort>=(rt_normalized_range(i)-rt_normalized_step) & normalizedrt_sort<=(rt_normalized_range(i)+rt_normalized_step))...
            ),2);
    end
    
    
    
    %     x = [0,1];
    y = [1, max(freq)];
    imagesc(x,y,temp) 
    xlabel(targetName)
    ylabel('freq (Hz)')
    % caxis([-3,-1.2])
    c = colorbar;
    c.Label.String = 'power';
    colormap('jet')
    
    title(['EEG ',componentMode,' power spectrum - sub',num2str(subInput)])
    
    %% 特殊设置
    if strcmp(componentMode,'osci')
        caxis([0,0.1])
        
    end
    
    
    %%
elseif strcmp(MethodMode,'meanSubEachPlot')
    
    count = 0;
    for subIndex = subInput
        count = count+1;
        
        subplot(ceil(sqrt(length(subInput))), ceil(sqrt(length(subInput))), count)
        
        eegspectrum_reshape = max(0,cat(3,eegspectrum{cat(2,subSet{subIndex})}));
        
        [normalizedrt_sort,sortIndex] = sort(cat(2, normalizedrt{cat(2,subSet{subIndex})}));
        
        if strcmp(componentMode,'mixd') || strcmp(componentMode,'frac')
            
            eegspectrumList_sort = log10(squeeze(mean(eegspectrum_reshape(:,:,sortIndex),1)));
            
        elseif strcmp(componentMode,'osci')
            
            eegspectrumList_sort = squeeze(mean(eegspectrum_reshape(:,:,sortIndex),1));
            
        end
        
        
        
        
        for i = 1:length(rt_normalized_range)
            temp(:,i) = mean(eegspectrumList_sort(:,...
                find(normalizedrt_sort>=(rt_normalized_range(i)-rt_normalized_step) & normalizedrt_sort<=(rt_normalized_range(i)+rt_normalized_step))...
                ),2);
        end
        
        
        
        %     x = [0,1];
        y = [1, max(freq)];
        imagesc(x,y,temp)
        xlabel(targetName)
        ylabel('freq (Hz)')
        % caxis([-3,-1.2])
        c = colorbar;
        c.Label.String = 'power';
        colormap('jet')
        
        title(['EEG ',componentMode,' power spectrum - sub',num2str(subIndex)])
        
        %% 特殊设置
        if strcmp(componentMode,'osci')
            caxis([0,0.1])
            
        end
        
        
    end
    
%%
elseif strcmp(MethodMode,'singleSubAllSession')
     count = 0;
    
    
    for sessionIndex = subSet{subInput}
        
        count = count + 1;
          
        subplot(2,3,count)
        
        eegspectrum_reshape = max(0,cat(3,eegspectrum{sessionIndex}));
        
        
        [normalizedrt_sort,sortIndex] = sort(cat(2, normalizedrt{sessionIndex}));
        
        
        if strcmp(componentMode,'mixd') || strcmp(componentMode,'frac')
            
            eegspectrumList_sort = log10(squeeze(mean(eegspectrum_reshape(:,:,sortIndex),1)));
            
        elseif strcmp(componentMode,'osci')
            
            eegspectrumList_sort = squeeze(mean(eegspectrum_reshape(:,:,sortIndex),1));
            
        end
        
        
        
        
        for i = 1:length(rt_normalized_range)
            temp(:,i) = mean(eegspectrumList_sort(:,...
                find(normalizedrt_sort>=(rt_normalized_range(i)-rt_normalized_step) & normalizedrt_sort<=(rt_normalized_range(i)+rt_normalized_step))...
                ),2);
        end
        
        
        
        %         x = [0,1];
        y = [1, max(freq)];
        imagesc(x,y,temp)
        xlabel(targetName),ylabel('freq (Hz)')
        % caxis([-3,-1.2])
        c = colorbar;
        c.Label.String = 'power';
        colormap('jet')
        %     title(['EEG ',componentMode,' power spectrum '])
        title(['session ',num2str(sessionIndex)])
    end
    
    suptitle(['EEG ',componentMode,' power spectrum - sub',num2str(subInput)])
    

elseif strcmp(MethodMode,'meanSubEachSession')
    
    %     subInput = varargin{3};
    count = 0;
    
    
    for sessionIndex = cat(2,subSet{subInput}) % subSet{subInput}
        
        count = count + 1;
        
%         subplot(ceil(sqrt(length(cat(2,subSet{subInput})))), ceil(sqrt(length(cat(2,subSet{subInput})))), count)
        subplot(6,6,sessionIndex)
        
        eegspectrum_reshape = max(0,cat(3,eegspectrum{sessionIndex}));
        
        
        [normalizedrt_sort,sortIndex] = sort(cat(2, normalizedrt{sessionIndex}));
        
        
        if strcmp(componentMode,'mixd') || strcmp(componentMode,'frac')
            
            eegspectrumList_sort = log10(squeeze(mean(eegspectrum_reshape(:,:,sortIndex),1)));
            
        elseif strcmp(componentMode,'osci')
            
            eegspectrumList_sort = squeeze(mean(eegspectrum_reshape(:,:,sortIndex),1));
            
        end
        
        
        
        
        for i = 1:length(rt_normalized_range)
            temp(:,i) = mean(eegspectrumList_sort(:,...
                find(normalizedrt_sort>=(rt_normalized_range(i)-rt_normalized_step) & normalizedrt_sort<=(rt_normalized_range(i)+rt_normalized_step))...
                ),2);
        end
        
        
        
        %         x = [0,1];
        y = [1, max(freq)];
        imagesc(x,y,temp)
        xlabel(targetName),ylabel('freq (Hz)')
        % caxis([-3,-1.2])
        c = colorbar;
        c.Label.String = 'power';
        colormap('jet')
        %     title(['EEG ',componentMode,' power spectrum '])
        subIndex = cellfun(@(x)find(x==sessionIndex),subSet,'UniformOutput',false);
        subIndex = find(cellfun(@isempty,subIndex)==0);
        title(['sub ',num2str(subIndex),' session ',num2str(sessionIndex)])
    end
    
    suptitle(['EEG ',componentMode,' power spectrum - sub',num2str(subInput)])
    
    
end



end

