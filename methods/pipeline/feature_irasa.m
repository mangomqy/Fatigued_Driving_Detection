classdef feature_irasa
    
    
    properties
        srate;
        hset;
        
        
        
    end
    %% �����������
    %     hset=1.1:0.05:1.65
    %     Ndata = 2^floor(log2(Ntotal*0.9)); % ���� 2048point
    %     nfft 4096 Ƶ�ʷֱ���0.12hz
    %     frange: separating[1, 50], fit[1, 50]
    %     simulation: �źŲ��� srate = 100hz
    %
    %  20200731
    % 1. irasa У��fft -> psd ʱ�� ()^2 -> ()^2/nfft
    % 2. fooof Ϊ��ֹ����ϣ��趨settings.max_n_peaks = 5��
    %
    % 20200820
    % 1. irasa���²���psd����ʱ����ϵ���޸�
    % 2. ����taiwan������ݼ�Ԥ��������1-50hz bandpass������fmax=50/hset(end)��Ϊ�˿���30hz��ȡhset=1.1:0.05:1.65
 
    
    
    methods
        function obj = feature_irasa(srate)
            %% defaults
            obj.srate = srate;
            obj.hset = 1.1:0.05:1.65;
            
        end
        
        
    end
    
    methods
        function [fea_output,freq] = separating_main(obj,data,frange,chanNum,varargin)
            % input: 
            %   data(channel, time point, trial)
            %   varargin{1}: �������� 'mixd'  or 'sep'
            %   varargin{2}: �Ƿ񷵻�����psd���ݣ���������Է��� 'classification' or
            %   'corranalysis' or 'both'
            
            for chanIndex = 1:chanNum
                
                sig = squeeze(data(chanIndex,:,:)); % sig:(timepoint, trial)
                
                
                % �������psd
                fracData = separating_input(obj, sig, frange,varargin{1});
                
                freq = fracData.freq;
                
                if strcmp(varargin{2},'classification')
                    
                    % ��ȡ����
                    fea = separating_output_extractfeature(obj, fracData,varargin{1});
                    
                    fea_output(chanIndex,:,:) = fea;

                    
                elseif strcmp(varargin{2},'corranalysis')
                    
                    % ��������psd�������ڷ��������
                    fea_output{chanIndex} = fracData;
                    
                elseif strcmp(varargin{2},'both')
                    % ��ȡ����
                    fea = separating_output_extractfeature(obj, fracData,varargin{1}); % dimension * trial
                    
                    fea_output.featureMatrix(chanIndex,:,:) = fea;
                    % var feature
%                     fea_output.featureMatrix(chanIndex,:,:) = var(sig);
                    
                    fea_output.allPSD{chanIndex} = fracData;
                    
                    
                    
                end
            end
        end
        
    end
    
    
    methods
        %% ��ں������������ݽṹ����
        function fracData = separating_input(obj, data, frange, componentMode) 
            % data:(timepoint, trial)
            %fracData:{x trial}.sturct ->(freq, chan)
            
            if ~isempty(data)
                
                if strcmp(componentMode,'sep') || strcmp(componentMode,'frac')
                    % seperate
                    spec = obj.amri_sig_fractal(data,'frange',frange); % �ṹ��
                    % fit
                    spec = obj.amri_sig_plawfit(spec,frange);
                    
                elseif strcmp(componentMode,'mixd') || strcmp(componentMode,'mixd_freqbins')
                    spec = obj.amri_sig_mixd(data,'frange',frange); % �ṹ��
                end
               
                fracData = spec;
              
            else
                fracData = [];
                
            end
        end
        
        %% ���ں������������������ȡ
        function fea = separating_output_extractfeature(obj, fracData, varargin)% freq point * trial
            
            freq = fracData.freq;
            
            bandfreqfrange =  {[1,4],[4,8],[8,13],[13,30]};
            
            MinPeakProminence = 1*10^(-3);
            
            
            bandfrange = {intersect(find(freq>=1),find(freq<4),'stable'),...
                intersect(find(freq>=4),find(freq<8),'stable'),...
                intersect(find(freq>=8),find(freq<13),'stable'),...
                intersect(find(freq>=13),find(freq<=30),'stable')};
            
            all = squeeze(mean(fracData.mixd(:,:),1));
            
            mixd_all = log10(squeeze(sum(fracData.mixd(:,:),1)));
            
            % 1. �����������
            for bandIndex = 1:length(bandfrange) 
                
%                1. ����������ƽ����
%                 feature_mixd(bandIndex,:) = log10(squeeze(mean(fracData.mixd(bandfrange{bandIndex},:),1))); % band, trial

%                2. �����������ͣ�
                feature_mixd(bandIndex,:) = log10(squeeze(sum(fracData.mixd(bandfrange{bandIndex},:),1))); % band, trial
                
                
%                3.  % �������
%                 feature_mixd(bandIndex,:) = squeeze(sum(fracData.mixd(bandfrange{bandIndex},:),1))./...
%                     squeeze(sum(fracData.mixd(:,:),1)); % band, trial
                
            end
            
%             for bandbinIndex = 1:29
%                 feature_mixd2(bandbinIndex,:) = log10(squeeze(mean(fracData.mixd(...
%                     intersect(find(freq>=bandbinIndex),find(freq<bandbinIndex+1),'stable')...
%                     ,:),1))); % band, trial
%             end
%             

            
            
            if strcmp(varargin{1},'sep')
                % 2. frac����
                feature_frac(1,:) = fracData.Beta;
                
                feature_frac(2,:) = fracData.Cons;
                

                
                % 3. osci��������
                osci = fracData.osci;
                
                try
                % ��ȡ��ֵ
                for trialIndex = 1:size(osci,2)
                    
                    [pks,locs, w, p] = findpeaks(osci(:,trialIndex),'MinPeakProminence',MinPeakProminence,'MinPeakDistance',4);
                    
                    % 1-ÿ��Ƶ����ȡ��ߵķ�:4*2������
                    for bandIndex = 1:length(bandfreqfrange)
                        oscipeakindex = intersect(find(freq(locs)<bandfreqfrange{bandIndex}(2)),...
                            find(freq(locs)>bandfreqfrange{bandIndex}(1)),'stable');
                        
                        %                 % 1-2: w p 2������
                        if ~isempty(oscipeakindex)
                            [~,pmaxindex] = max(p(oscipeakindex));
                            feature_osci(1+(bandIndex-1)*2,trialIndex) = ...
                                w(oscipeakindex(pmaxindex));
                            feature_osci((bandIndex)*2,trialIndex) = ...
                                p(oscipeakindex(pmaxindex));
                        else
                            feature_osci(1+(bandIndex-1)*2,trialIndex) = 0; %w
                            feature_osci((bandIndex)*2,trialIndex) = 0; %p
                            
                        end
                        
                        
                    end
                    
                end
                catch
                    s=1;
                end
                
                count = 0;
                for i = 1:2:length(freq)
                    count = count+1;
                    mixd_power_1hz_bin(count,:) = mean(fracData.mixd(i:min(i+1,end),:),1);
                end
                
                % 4+2+8
%                 fea = cat(1,feature_mixd,feature_frac,feature_osci);
                fea = cat(1,feature_mixd,feature_frac);
%                 fea = cat(1,feature_frac);
                
                % 4+2+8+30
%                 fea = cat(1,feature_mixd,feature_frac,feature_osci,log10(mixd_power_1hz_bin));
                
                % 59+59+59+2
%                  fea = cat(1,fracData.mixd,fracData.frac,fracData.osci,feature_frac);
                
                
                % 
            elseif strcmp(varargin{1},'frac')
                
                % 2. frac����
                feature_frac(1,:) = fracData.Beta;
                
                feature_frac(2,:) = fracData.Cons;
                
                fea = feature_frac;
                
            elseif strcmp(varargin{1},'mixd')
                %% ��ȡ��Ϲ���������
                % ��4��Ƶ��
                fea = cat(1,feature_mixd,mixd_all);
%                 fea = feature_mixd;
               
                % freq bins [1-30hz]
%                 fea = feature_mixd2;
                
                % de����
%                 fea = fea;
            elseif strcmp(varargin{1},'mixd_freqbins')
                
                fea = log10(fracData.mixd);
%                 fea = fracData.mixd;
                
            end
                
            
        end
        

    end
    
      %% ��psd����: pwelch
    methods 
        function spec = amri_sig_mixd(obj,sig,varargin)
            %% defaults
            flag_detrend = 1;  
            flag_filter = 1;
            fmin = 0;
            fmax = obj.srate/2;
            
             %% Keywords
            for i = 1:2:size(varargin,2)
                Keyword = varargin{i};
                Value   = varargin{i+1};
                if strcmpi(Keyword,'frange')
                    fmin = max(Value(1),fmin);
                    fmax = min(Value(2),fmax);
                elseif strcmpi(Keyword,'detrend')
                    flag_detrend = Value;
                elseif strcmpi(Keyword,'filter')
                    flag_filter = Value;
                elseif strcmpi(Keyword,'hset')
                    obj.hset = Value;
                else
                    warning(['amri_sig_mixd(): unknown keyword ' Keyword]);
                end
            end
            
            %% preprocessing
            sig = double(sig);
            if isvector(sig)
                sig = sig(:);
            end
            
            % detrend signal
            if flag_detrend >= 1
                sig = detrend(sig,'linear');
            end
            
            %% apply pwelch method: sig (timepoint * trial)
%             for i = 1:size(sig,2)
%                 
%                 
%             end
            % Ĭ��hamming��
            Ntotal = size(sig,1);
            dim = size(sig,2);
            
            % Ndata is the power of 2 that does not exceed 90% of Ntotal.
            Ndata = 256; % ����250hz����������
            
            % Nsubset is fixed to 15
            Nsubset = 15;
            
            % compute the auto-power spectrum of the originally sampled time series
            L = floor((Ntotal-Ndata)/(Nsubset-1));
            
            % set nfft greater than ceil(hset(end))*Ndata, asure that do fft without truncating
            nfft = 2^nextpow2(ceil(obj.hset(end))*Ndata);
            
            % set output data length Nfrac
            Nfrac = nfft/2 + 1;
            
  
            [pxx, freq] = pwelch(sig, Ndata, L, nfft, obj.srate);
            
            Smixd = pxx;

            
            
            
            %% only keep the given frequency range
            ff = (freq>=fmin & freq<=fmax & freq>0);
            freq = freq(ff);
            Smixd = Smixd(ff,:);
            
            %% outputs
            spec.freq  = freq;
            spec.srate = obj.srate;
            spec.mixd  = Smixd;
            spec.frac  = 0;
            spec.osci  = 0;
            
            
        end
        
    end
    
    %% irasa
    methods
        % Separate the spectra of fractal component and oscillatory component from mixed time series
        function spec = amri_sig_fractal(obj,sig,varargin)
            
            if nargin<2
                eval('help amri_sig_fractal');
                return
            end
            
            %% defaults
            flag_detrend = 1;  
            flag_filter = 1;
            fmin = 0;
            fmax = obj.srate/2/obj.hset(end);
            
            %% Keywords
            for i = 1:2:size(varargin,2)
                Keyword = varargin{i};
                Value   = varargin{i+1};
                if strcmpi(Keyword,'frange')
                    fmin = max(Value(1),fmin);
                    fmax = min(Value(2),fmax);
                elseif strcmpi(Keyword,'detrend')
                    flag_detrend = Value;
                elseif strcmpi(Keyword,'filter')
                    flag_filter = Value;
                elseif strcmpi(Keyword,'hset')
                    obj.hset = Value;
                else
                    warning(['amri_sig_fractal(): unknown keyword ' Keyword]);
                end
            end
            
            %% preprocessing
            sig = double(sig);
            if isvector(sig)
                sig = sig(:);
            end
            
            % detrend signal
            if flag_detrend >= 1
                sig = detrend(sig,'linear');
            end
            
            %% apply IRASA method to separate fractal and oscillatory components
            [Smixd, Sfrac, freq] = obj.irasa(sig,flag_filter);
            
            %% only keep the given frequency range
            ff = (freq>=fmin & freq<=fmax & freq>0);
            freq = freq(ff);
            Smixd = Smixd(ff,:);
            Sfrac = Sfrac(ff,:);
            
            %% outputs
            spec.freq  = freq;
            spec.srate = obj.srate;
            spec.mixd  = Smixd;
            spec.frac  = Sfrac;
            spec.osci  = Smixd - Sfrac;
            
        end     
        
        % IRASA Irregular-Resampling Auto-Spectral Analysis
        function [Smixd, Sfrac, freq] = irasa(obj,sig,flag_filter)
            % Given a discrete time series (sig) of length (Ntotal)
            Ntotal = size(sig,1);
            dim = size(sig,2);
            
            % Ndata is the power of 2 that does not exceed 90% of Ntotal.
            Ndata = 256; % ����250hz����������
%             Ndata = 2^floor(log2(Ntotal*0.9)); 
            
            % Nsubset is fixed to 15
            Nsubset = 15;
            
            % compute the auto-power spectrum of the originally sampled time series
            L = floor((Ntotal-Ndata)/(Nsubset-1));
            
            % set nfft greater than ceil(hset(end))*Ndata, asure that do fft without truncating
            nfft = 2^nextpow2(ceil(obj.hset(end))*Ndata);
            
            % set output data length Nfrac
            Nfrac = nfft/2 + 1;
            freq = obj.srate/2*linspace(0,1,Nfrac); freq = freq(:);
            
            %% compute the spectrum of mixed data
            Smixd = zeros(Nfrac,dim);
            taper = obj.gettaper([Ndata dim]);
            for k = 0:Nsubset-1
                i0 = L*k+1;
                x1 = sig(i0:1:i0+Ndata-1,:);
                p1 = fft(1.633*x1.*taper,nfft)/min(nfft,size(x1,1)); %/N ����У����hann��ϵ��1.633��
                p1(2:end,:) = p1(2:end,:)*2; % ������
                Smixd = Smixd+abs(p1(1:Nfrac,:)).^2/min(nfft,size(x1,1));%
            end
            Smixd = Smixd/Nsubset; % ƽ��mixed������
            
            
            
            %% filter the input signal to avoid alising when downsampling ��LP���ٳ�ȡ��
            if flag_filter == 1
                sig_filtered = sig;
                for i = 1 : size(sig,2)
                    sig_filtered(:,i) = obj.amri_sig_filtfft(sig(:,i),obj.srate,0,obj.srate/(2*ceil(obj.hset(end))));
                end
            end
            
            % compute fractal component.
            Sfrac = zeros(Nfrac,dim,length(obj.hset));
            for ih = 1:length(obj.hset)
                %% compute the auto-power spectrum of xh �ϲ���
                h = obj.hset(ih);
                [n, d] = rat(h); % n > d
                Sh = zeros(Nfrac,dim);
                for k = 0 : Nsubset-1
                    i0 = L*k + 1;
                    x1 = sig(i0:i0+Ndata-1,:);
                    xh = obj.myresample(x1, n, d); % ���Բ�ֵ
                    taperh = obj.gettaper(size(xh));
                    ph = fft(1.633*xh.*taperh,nfft)/min(nfft,size(xh,1));%�Ӵ�fft
                    ph(2:end,:) = ph(2:end,:)*2;
                    tmp = (abs(ph)).^2;
                    Sh = Sh + tmp(1:Nfrac,:)/min(nfft,size(xh,1)); % ��ȡ����� 
                end
                Sh = Sh / Nsubset;
                Sh_temp(:,:,ih) = Sh;
               
                %     Sh = Sh / Ndata; % ʱ��ƽ��-psd
                
                %% compute the auto-power spectrum of X1h ������
                S1h = zeros(Nfrac, dim);
                for k = 0 : Nsubset - 1
                    i0 = L*k + 1;
                    if (flag_filter==1)
                        x1 = sig_filtered(i0:1:i0+Ndata-1,:);
                    else
                        x1 = sig(i0:1:i0+Ndata-1,:);
                    end
                    x1h = obj.myresample(x1,d,n);
                    taper1h = obj.gettaper(size(x1h));
                    p1h = fft(1.633*x1h.*taper1h,nfft)/min(nfft,size(x1h,1));
                    p1h(2:end,:) = p1h(2:end,:)*2;
                    tmp = (abs(p1h)).^2;
                    S1h = S1h + tmp(1:Nfrac,:)/min(nfft,size(x1h,1));%
                end
                S1h = S1h / Nsubset;
                %     S1h = S1h / Ndata; % ʱ��ƽ��-psd
                
                Sfrac(:,:,ih)= sqrt(Sh.*S1h);
            end
            

            % taking median ÿ��Ƶ�ʵ㴦ȡ��λ��
            Sfrac = median(Sfrac,3);

            
        end
           
        % fitting power-law function to scale-free power-spectrum
        function spec = amri_sig_plawfit(obj, spec, frange)
            
            % define frequency range
            ff = spec.freq >= frange(1) & spec.freq <= frange(2);
            freq = spec.freq(ff);
            frac = spec.frac(ff,:,:);
            
            % convert to log-log scale
            logfreq = log10(freq);
            y1 = log10(frac);
            
            % resample frac in equal space
            x2 = linspace(min(logfreq),max(logfreq),length(logfreq)); x2 = x2(:);
            y2 = interp1(logfreq,y1,x2);
            
            % fitting power-law function
            Nt = size(y2,2);
            Nc = size(y2,3);
            beta = zeros(Nt,Nc);
            cons = zeros(Nt,Nc);
            plaw = zeros(size(frac));
            
            for j = 1 : Nc
                for i = 1 : Nt
                    % ordinary least square
                    p = polyfit(x2,y2(:,i,j),1);
                    
                    beta(i,j) = -p(1);
                    cons(i,j) = p(2);
                    powlaw = 10.^(polyval(p,logfreq));
                    plaw(:,i,j) = powlaw(:);
                end
            end
            
            % outputs
            spec.Beta = beta;
            spec.Cons = cons;
            spec.Plaw = plaw;
            spec.Freq = freq;
            
        end

        %
        function ts_new = amri_sig_filtfft(obj,ts, fs, lowcut, highcut, revfilt, trans)
            
            if nargin<1
                eval('help amri_sig_filtfft');
                return
            end
            
            if ~isvector(ts)
                printf('amri_sig_filtfft(): input data has to be a vector');
            end
            
            if nargin<2,fs=1;end                % default sampling frequency is 1 Hz, if not specified
            if nargin<3,lowcut=NaN;end          % default lowcut is NaN, if not specified
            if nargin<4,highcut=NaN;end         % default highcut is NaN, if not specified
            if nargin<5,revfilt=0;end           % default revfilt=0: bandpass filter
            if nargin<6,trans=0.15;end          % default relative trans of 0.15
            
            [ts_size1, ts_size2] = size(ts);    % save the original dimension of the input signal
            ts=ts(:);                           % convert the input into a column vector
            npts = length(ts);                  % number of time points
            nfft = 2^nextpow2(npts);            % number of frequency points
            
            fv=fs/2*linspace(0,1,nfft/2+1);     % even-sized frequency vector from 0 to nyguist frequency
            fres=(fv(end)-fv(1))/(nfft/2);      % frequency domain resolution
            % fv=fs/2*linspace(0,1,nfft/2);     % even-sized frequency vector from 0 to nyguist frequency
            % fres=(fv(end)-fv(1))/(nfft/2-1);  % frequency domain resolution
            
            
            filter=ones(nfft,1);                % desired frequency response
            
            % remove the linear trend
            ts_old = ts;
            ts = detrend(ts_old,'linear');
            trend  = ts_old - ts;
            
            % design frequency domain filter
            if (~isnan(lowcut)&&lowcut>0)&&...          % highpass
                    (isnan(highcut)||highcut<=0)
                
                %          lowcut
                %              -----------
                %             /
                %            /
                %           /
                %-----------
                %    lowcut*(1-trans)
                
                idxl = round(lowcut/fres)+1;
                idxlmt = round(lowcut*(1-trans)/fres)+1;
                idxlmt = max([idxlmt,1]);
                filter(1:idxlmt)=0;
                filter(idxlmt:idxl)=0.5*(1+sin(-pi/2+linspace(0,pi,idxl-idxlmt+1)'));
                filter(nfft-idxl+1:nfft)=filter(idxl:-1:1);
                
            elseif (isnan(lowcut)||lowcut<=0)&&...      % lowpass
                    (~isnan(highcut)&&highcut>0)
                
                %        highcut
                % ----------
                %           \
                %            \
                %             \
                %              -----------
                %              highcut*(1+trans)
                
                idxh=round(highcut/fres)+1;
                idxhpt = round(highcut*(1+trans)/fres)+1;
                filter(idxh:idxhpt)=0.5*(1+sin(pi/2+linspace(0,pi,idxhpt-idxh+1)'));
                filter(idxhpt:nfft/2)=0;
                filter(nfft/2+1:nfft-idxh+1)=filter(nfft/2:-1:idxh);
                
            elseif lowcut>0&&highcut>0&&highcut>lowcut
                if revfilt==0                           % bandpass (revfilt==0)
                    
                    %         lowcut   highcut
                    %             -------
                    %            /       \     transition = (highcut-lowcut)/2*trans
                    %           /         \    center = (lowcut+highcut)/2;
                    %          /           \
                    %   -------             -----------
                    % lowcut-transition  highcut+transition
                    transition = (highcut-lowcut)/2*trans;
                    idxl   = round(lowcut/fres)+1;
                    idxlmt = round((lowcut-transition)/fres)+1;
                    idxh   = round(highcut/fres)+1;
                    idxhpt = round((highcut+transition)/fres)+1;
                    idxl = max([idxlmt,1]);
                    idxlmt = max([idxlmt,1]);
                    idxh = min([nfft/2 idxh]);
                    idxhpt = min([nfft/2 idxhpt]);
                    filter(1:idxlmt)=0;
                    filter(idxlmt:idxl)=0.5*(1+sin(-pi/2+linspace(0,pi,idxl-idxlmt+1)'));
                    filter(idxh:idxhpt)=0.5*(1+sin(pi/2+linspace(0,pi,idxhpt-idxh+1)'));
                    filter(idxhpt:nfft/2)=0;
                    filter(nfft-idxl+1:nfft)=filter(idxl:-1:1);
                    filter(nfft/2+1:nfft-idxh+1)=filter(nfft/2:-1:idxh);
                    
                else                                    % bandstop (revfilt==1)
                    
                    % lowcut-transition  highcut+transition
                    %   -------             -----------
                    %          \           /
                    %           \         /    transition = (highcut-lowcut)/2*trans
                    %            \       /     center = (lowcut+highcut)/2;
                    %             -------
                    %         lowcut   highcut
                    
                    
                    transition = (highcut-lowcut)/2*trans;
                    idxl   = round(lowcut/fres)+1;
                    idxlmt = round((lowcut-transition)/fres)+1;
                    idxh   = round(highcut/fres)+1;
                    idxhpt = round((highcut+transition)/fres)+1;
                    idxlmt = max([idxlmt,1]);
                    idxlmt = max([idxlmt,1]);
                    idxh = min([nfft/2 idxh]);
                    idxhpt = min([nfft/2 idxhpt]);
                    filter(idxlmt:idxl)=0.5*(1+sin(pi/2+linspace(0,pi,idxl-idxlmt+1)'));
                    filter(idxl:idxh)=0;
                    filter(idxh:idxhpt)=0.5*(1+sin(-pi/2+linspace(0,pi,idxl-idxlmt+1)'));
                    filter(nfft-idxhpt+1:nfft-idxlmt+1)=filter(idxhpt:-1:idxlmt);
                    
                end
                
            else
                printf('amri_sig_filtfft(): error in lowcut and highcut setting');
            end
            
            X=fft(ts,nfft);                         % fft
            ts_new = real(ifft(X.*filter,nfft));    % ifft
            ts_new = ts_new(1:npts);                % tranc
            
            % add back the linear trend
            ts_new = ts_new + trend;
            
            ts_new = reshape(ts_new,ts_size1,ts_size2);
            
            return
        end
        
        % Simulate fractal time series generating function
        function frac = amri_sig_genfrac(obj, N,Nfft,varargin)
            if nargin<2
                eval('help amri_sig_genfrac');
                return
            end
            
            %% Defaults
            thetarange = [0 2*pi];
            beta = 1;
            cons = 1;
            
            %% Keywords
            for i = 1:2:size(varargin,2)
                Keyword = varargin{i};
                Value   = varargin{i+1};
                if strcmpi(Keyword,'theta')
                    thetarange = Value;
                elseif strcmpi(Keyword,'beta')
                    beta = Value;
                elseif strcmpi(Keyword,'cons')
                    cons = Value;
                else
                    warning(['amri_sig_genfrac(): unknown keyword ' Keyword]);
                end
            end
            
            %% generate fractal time series��
            % ��freq���н�ģ��cons*freq.^(-beta) ����freq = 0��ֱ�������Ĺ���
            t = 1 : N; % ʱ��
            k = (1 : Nfft/2-1)'; % Ƶ�򣺵���fft
            kt = k*t;
            % ��λ���
            srate = 100;
            freq = srate/2*linspace(0,1,Nfft/2);
            
            freq = freq(2:end)';
            
            theta = unifrnd(thetarange(1),thetarange(2),Nfft/2-1,1);
            Ck = repmat(sqrt(cons*freq.^(-beta)*Nfft),1,N(1)); % ck��Nfft,N��: fft�����psd: cons*k.^(-beta)
            frac = sum(Ck.*cos(2*pi*kt/Nfft - repmat(theta,1,N)));%/Nfft
            frac = frac(:);
            
            %             % ֱ�Ӷ�predefined-psd�������
            %             freq = freq(2:end);
            %             P = cons*freq.^(-beta); % ��freqΪ�������ʲô���⣿��һ������inf
            %             spec.frac = P; % ��kΪ����
            %             spec.freq = freq;
            %             spec = obj.amri_sig_plawfit(spec, [0.1,30]);
            %             betaHat = spec.Beta;
            %             % һ�����
            %             consHat = 10^(spec.Cons);
            %             % ����
            % %             consHat = 10^(spec.Cons)*(srate/Nfft)^(-betaHat);
            %
            %
            %             fprintf(['Estimated: beta - ',num2str(betaHat),', cons - ',num2str(consHat)]);
            
            
            %             % plot psd
            %             figure(),
            %             loglog(freq,P,'LineWidth',2), hold on %cons*k.^(-beta)
            %             loglog(spec.Freq,spec.Plaw,'LineWidth',2), hold on
            %             legend('cons*k.^(-beta)','cons*k.^(-beta)~estimated')
            %             title('Theoretical psd of frac comp')
            %             xlim([0,30])
            %
            
            
        end

    end
    
    

    
    
    %% fooof
    methods
        function [Smixd, fooof_results] = fit_fooof(obj,sig,srate,frange)
            %% ���㹦����
            % Given a discrete time series (sig) of length (Ntotal)
            Ntotal = size(sig,1);
            dim = size(sig,2);
            
            % Ndata is the power of 2 that does not exceed 90% of Ntotal.
            Ndata = 2^floor(log2(Ntotal*0.2)); % ���� 512 point
            % Ndata = 512; % ���� 512 point
            
            % Nsubset is fixed to 15
            Nsubset = 15;
            
            % compute the auto-power spectrum of the originally sampled time series
            L = floor((Ntotal-Ndata)/(Nsubset-1));
            
            % set nfft greater than ceil(hset(end))*Ndata, asure that do fft without truncating
            %             nfft = 2^nextpow2(ceil(hset(end))*Ndata);
            nfft = Ndata;
            
            % set output data length Nfrac
            Nfrac = nfft/2 + 1;
            freq = srate/2*linspace(0,1,Nfrac); freq = freq(:);
            
            %% compute the spectrum of mixed data
            Smixd = zeros(Nfrac,dim);
            taper = obj.gettaper([Ndata dim]);
            for k = 0:Nsubset-1
                i0 = L*k+1;
                x1 = sig(i0:1:i0+Ndata-1,:);
                p1 = fft(1.633*x1.*taper,nfft)/min(nfft,size(x1,1)); %/N ����У����hann��ϵ��1.633��
                p1(2:end,:) = p1(2:end,:)*2; % ������
                Smixd = Smixd+abs(p1(1:Nfrac,:)).^2/min(nfft,size(x1,1));%/min(nfft,size(x1,1))
            end
            Smixd = Smixd/Nsubset; % ƽ��mixed������
            
            freq = freq';
            psd = Smixd';
            
            % FOOOF settings
            settings = struct();  % Use defaults
            settings.max_n_peaks = 5;
            
            %             frange = [1, 30];
            
            % Run FOOOF
            % aperiodic_params (offset, exponent)
            fooof_results = fooof(freq, psd, frange, settings,true);
            
        end
        
        function [Smixd, fooof_results] = fit_fooof_group(obj,sig,srate,frange)
            %% ���㹦����
            % Given a discrete time series (sig) of length (Ntotal)
            Ntotal = size(sig,1);
            dim = size(sig,2);
            
            % Ndata is the power of 2 that does not exceed 90% of Ntotal.
            Ndata = 2^floor(log2(Ntotal*0.2)); % ���� 512 point
            % Ndata = 512; % ���� 512 point
            
            % Nsubset is fixed to 15
            Nsubset = 15;
            
            % compute the auto-power spectrum of the originally sampled time series
            L = floor((Ntotal-Ndata)/(Nsubset-1));
            
            % set nfft greater than ceil(hset(end))*Ndata, asure that do fft without truncating
            %             nfft = 2^nextpow2(ceil(hset(end))*Ndata);
            nfft = Ndata;
            
            % set output data length Nfrac
            Nfrac = nfft/2 + 1;
            freq = srate/2*linspace(0,1,Nfrac); freq = freq(:);
            
            %% compute the spectrum of mixed data
            Smixd = zeros(Nfrac,dim);
            taper = obj.gettaper([Ndata dim]);
            for k = 0:Nsubset-1
                i0 = L*k+1;
                x1 = sig(i0:1:i0+Ndata-1,:);
                p1 = fft(1.633*x1.*taper,nfft)/min(nfft,size(x1,1)); %/N ����У����hann��ϵ��1.633��
                p1(2:end,:) = p1(2:end,:)*2; % ������
                Smixd = Smixd+abs(p1(1:Nfrac,:)).^2/min(nfft,size(x1,1));%/min(nfft,size(x1,1))
            end
            Smixd = Smixd/Nsubset; % ƽ��mixed������
            
            freq = freq';
            psd = Smixd;
            
            % FOOOF settings
            settings = struct();  % Use defaults
            settings.max_n_peaks = 5;
            %             frange = [1, 30];
            
            % Run FOOOF
            % aperiodic_params (offset, exponent)
            fooof_results = fooof_group(freq, psd, frange, settings);
            % freq 1,257; psd 58*257
        end
        
    end
    
    %% subfunctions
    methods
        function taper = gettaper(obj,S)
            % get a tapering function for power spectrum density calculation
            % hann
            if license('test','signal_toolbox')
                taper = hann(S(1),'periodic');
            else
                taper = 0.5*(1-cos(2*pi*(1:S(1))/(S(1)-1)));
            end
            % %% kaiser
            %     taper = kaiser(S(1),9);
            taper = taper(:);
            taper = repmat(taper,1,S(2));
        end
        
        function data_out = myresample(obj,data,L,D)
            % resample signal with upsample L and downsample D
            if license('test','signal_toolbox')
                data_out = resample(data,L,D);
            else
                N = size(data,1);
                x0 = linspace(0,1,N);
                x1 = linspace(0,1,round(N*L/D));
                data_out = interp1(x0,data,x1);
            end
        end
        
    end
end

