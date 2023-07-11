% Echo combiantion for line-scanning data (without phase encoding gradient)
% Inputs:
% -data are data in the form x,time,echos 
% -echotimes is a vector containing the echo times
% -weghting can be chose as CNR for Poser 2006 echo combination or t2star


function output = CombineEchoes_nofor_LS(data,echotimes, weighting)
data = squeeze(data);
echo_dim = length(size(data));
time_dim = length(size(data))-1; 

data_Tmean=mean(abs(data),time_dim); 
data_Tstd=std(abs(data),[],time_dim);

if strcmp(weighting,'CNR')
    % CNR weighting, Poser MRM 2006
    % compute temporal SNR
     SNR=data_Tmean./data_Tstd;
     %figure, plot(squeeze(abs(SNR))); title('tSNR per echo'); 
     echotimes = reshape(echotimes, [ones(1,length(size(data))-1), length(echotimes)]);
     den = sum(echotimes.*SNR, echo_dim);
     w=echotimes.*SNR./den;
     %figure, plot(squeeze(abs(w))); title('weightings'); 
     output = sum(bsxfun(@times,abs(data),w),echo_dim);
end

if strcmp(weighting,'T2star')
    
[T2s_fit, S0_fit] = T2s_Fit_LS(abs(data),echotimes);

S0_fit=S0_fit;
T2s_fit=T2s_fit*1000;% to ms

%compute the normalization factor eq.6, Poser et al 2006
for n=1:length(echotimes)
    EXPT2STE(:,:,n)=echotimes(n)*exp(-echotimes(n)./T2s_fit);
end
w=EXPT2STE./sum(EXPT2STE,echo_dim);

output = squeeze(sum(bsxfun(@times,abs(data),w),echo_dim));  
    
end

if strcmp(weighting,'T2star_fit')
    
[T2s_fit, S0_fit] = T2s_Fit_LS(abs(data),echotimes);

S0_fit=S0_fit;
T2s_fit=T2s_fit*1000;% to ms

output = T2s_fit;  
    
end

if strcmp(weighting,'average')
    output = mean(data,echo_dim);
end

if strcmp(weighting,'SoS')
    output=sqrt(sum(abs(data),echo_dim).^2);
end
    
if strcmp(weighting,'complex')
    %Phase is now deltaB0 from the phase fit
    TEfix = echotimes(1); 
     phase_data = jump_correction(data);
     [deltaB0map, Phi0map] = phase_fit(phase_data,echotimes);
     gamma = 42.58*10^6; %Hz/T
     phase_rad = deltaB0map *gamma *TEfix; %phase is taken from the phase fit
     magn = sqrt(sum(abs(data),echo_dim).^2); %magnitude is the SoS of the values %    
     
%      echotimes = reshape(echotimes, [ones(1,length(size(data))-1), length(echotimes)]);    
%      SNR=data_Tmean./data_Tstd;  
%      den = sum(echotimes.*SNR, echo_dim);
%      %w=exp(-i*(phase_data-Phi0map)).*echotimes.*SNR./den;
%      w=exp(i*B0map).*echotimes.*SNR./den;
%      w=echotimes.*SNR./den;

     output = magn.*exp(i*phase_rad);

end

