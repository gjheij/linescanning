%% Reconstruction for line scanning, choose 1 way
% output: LS reconstructed data 
% inputs: 
%         data_lab = data in lab format
%         filelocation = path with the location of your data
%         Nc = total number of channels
%         TR = repetition time in ms
%         kind = can be 'all' if you want to reconstruct using all
%         channels, or 'some' if you wan to use only the channels which
%         contribute the most
%         gausskernel = FWHM of the gaussian for the smoothing kernel
%         (standard is 24)
%         kfilter = filter in kspace used to get rid of stripy artifacts
%         in k-space. can be 'yes' and 'no'
%         reconType = can be 'SoS' = Sum of squares (SoS) reconstruction
%         'wcsm' = complex reconstruction with coil sensitivity maps (csm)
%         'wtSNR' = complex recosntruction with tSNR per channel
%         'wcsmtSNR' = complex recosntruction with tSNR/channel and csm\
%         NORDIC = can be 'on' and 'off', depending if you want to perorm
%         nordic denoising
%         varargin = is a cell array. First element is Data_py = complex data with phase encoding on and
%         rest slabs, for every channel to evaluate coil sensitivity maps
%         and second argument of the cell array is the NORDIC treshold used in case NORDIC is on

function recon = LS_recon_1way(nopy_lab, filelocation, Nc, kind, TR, gausskernel, kfilter, reconType, NORDIC, varargin)

Snopy=MRecon([filelocation nopy_lab]);
Snopy.Parameter.Parameter2Read.typ = 1;
Snopy.ReadData;
Snopy.PDACorrection;
Snopy.RandomPhaseCorrection;
Snopy.RemoveOversampling;
Snopy.DcOffsetCorrection;
Snopy.MeasPhaseCorrection;
Snopy.SortData;
Snopy.PartialFourier;

dyn = 1;
figure, montage(squeeze(abs(Snopy.Data(:,:,1,:,dyn)))); caxis([0 10]); title ('k-space all channels')


if strcmp(kfilter, 'yes')
    disp ('k filter applyed')
    D = HammingFilter4LS(Snopy.Data); %this is a filter, removing the "stripes" in k-space, apparently
    Snopy.Data = D;
    figure, montage(squeeze(abs(D(:,:,1,:,dyn)))); caxis([0 10]); title ('k-space all channels after Hamming filter')

elseif strcmp(kfilter, 'no')
      disp ('no k filter applyed')
end

Snopy.K2IM; % fft in readout direction

%Snopy.RemoveOversampling;
dimsSnopy=size(Snopy.Data);
size(Snopy.Data)

% 1 dynamic
if length(dimsSnopy) == 4
    Data_nopy=squeeze(abs(Snopy.Data));
    Data_nopy_cp=squeeze(Snopy.Data);
else
%more dynamics
    Snopy.Data = flip(Snopy.Data,2); %each dynamic has time reversed!!! You correct fot that here
    Data_nopy=reshape(permute(abs(squeeze(Snopy.Data)), [1, 2, 4, 3]), dimsSnopy(1),dimsSnopy(2)*dimsSnopy(5),dimsSnopy(4));
    Data_nopy_cp=reshape(permute(squeeze(Snopy.Data), [1, 2, 4, 3]), dimsSnopy(1),dimsSnopy(2)*dimsSnopy(5),dimsSnopy(4));
end

%% NORDIC 
if strcmp(NORDIC,'on')
    disp ('NORDIC on')
    for ch =1:Nc
        %Data_nopy_nordic(:,:,ch) = nordic_psr(varargin{1,1}{2}, Data_nopy_cp(:,:,ch));
        Data_nopy_nordic(:,:,ch) = nordic_ev(Data_nopy_cp(:,:,ch));
    end
    Data_nopy_cp = Data_nopy_nordic;
else
    disp ('NORDIC off')
end

%% Reconstuction: different ways
% all channels
if strcmp(kind, 'all')
             
    if strcmp(reconType, 'SoS')
    %SOS
    Data_SoS=sqrt(sum(abs(Data_nopy_cp),3).^2);
    recon = Data_SoS;
    end
     if strcmp(reconType, 'wcsm')
    Data_py = varargin{1,1}{1};
    % Coil Sesnsitivity maps
    imagval = imag(Data_py);
    realval = real(Data_py);
    absval = abs(Data_py);
    phaseval = angle(Data_py);
    %gausskernel=24; standard
    norm = sqrt(sum(absval,3).^2);

    %compute Coil sensitivity maps
    csm_sm_cplx = complex(imgaussfilt(realval,gausskernel,'FilterDomain','frequency', 'Padding',0 ), imgaussfilt(imagval,gausskernel,'FilterDomain','frequency','Padding',0 ))./imgaussfilt(norm,gausskernel,'FilterDomain','frequency','Padding',0);
    csm_nosm_cplx = Data_py./norm;
    %compute Coil sensitivity lines
    %line_range=330:430;
    line_range = 1:size(csm_nosm_cplx,2);

    csm4ls_sm_cplx = squeeze(sum(csm_sm_cplx(:,line_range,:),2)); % check out range to sum over lines in regions with high SNR, JCWS
    csm4ls_nosm = squeeze(sum(csm_nosm_cplx(:,line_range,:),2));
    %Csm weighted
    for t=1:size(Data_nopy,2)
    Data_csm2(:,t,:) = sum(squeeze(Data_nopy_cp(:,t,:)).*conj(csm4ls_sm_cplx),2)./sqrt(sum(abs(csm4ls_sm_cplx).^2,2));
    end
    Data_csm_abs = abs(Data_csm2);
    recon = Data_csm2;
     end
    if strcmp(reconType, 'wtSNR')
    %wtSNR
    tSNRperCoil=squeeze(mean(abs(Data_nopy_cp),2)./std(abs(Data_nopy_cp),[],2)); %tSNR per coil
    for t=1:size(Data_nopy_cp,2)
    Data_wtSNR_num(:,t,:) = squeeze(Data_nopy_cp(:,t,:)).*tSNRperCoil;
    end
    Data_wtSNR=sum(Data_wtSNR_num,3)./sqrt(sum(tSNRperCoil,2).^2);
    Data_wtSNR_abs = abs(Data_wtSNR);
    recon = Data_wtSNR;
    end

    if strcmp(reconType, 'wcsmtSNR')
    %csm+tSNR weighted
    Data_py = varargin{1,1}{1};
    % Coil Sesnsitivity maps
    imagval = imag(Data_py);
    realval = real(Data_py);
    absval = abs(Data_py);
    phaseval = angle(Data_py);
    %gausskernel=24; standard
    norm = sqrt(sum(absval,3).^2);

    %compute Coil sensitivity maps
    csm_sm_cplx = complex(imgaussfilt(realval,gausskernel,'FilterDomain','frequency', 'Padding',0 ), imgaussfilt(imagval,gausskernel,'FilterDomain','frequency','Padding',0 ))./imgaussfilt(norm,gausskernel,'FilterDomain','frequency','Padding',0);
    csm_nosm_cplx = Data_py./norm;
    %compute Coil sensitivity lines
    %line_range=330:430;
    line_range = 1:size(csm_nosm_cplx,2);

    csm4ls_sm_cplx = squeeze(sum(csm_sm_cplx(:,line_range,:),2)); % check out range to sum over lines in regions with high SNR, JCWS
    csm4ls_nosm = squeeze(sum(csm_nosm_cplx(:,line_range,:),2));
    tSNRperCoil=squeeze(mean(abs(Data_nopy_cp),2)./std(abs(Data_nopy_cp),[],2)); %tSNR per coil
    
    for t=1:size(Data_nopy,2)
    Data_wcsmtSNR(:,t,:) = sum(squeeze(Data_nopy_cp(:,t,:)).*conj(csm4ls_sm_cplx).*tSNRperCoil,2)./sqrt(sum(abs(csm4ls_sm_cplx.*tSNRperCoil).^2,2));
    end
    Data_wcsmtSNR_abs = abs(Data_wcsmtSNR);
    recon = Data_wcsmtSNR;
    end
%some channels
elseif strcmp(kind, 'some')
    
%     mean_csm = abs(mean(csm4ls_sm_cplx)); %if you select coils based on csm 
%     useful_coils = find(mean_csm>mean(mean_csm)*2.7);
    
    max_coils = abs(max(max(max(Data_py,3)))); %if you select coils based on the image
    useful_coils = find(max_coils>max(max_coils)/2);

    for c=1:Nc
        for i=1:length(useful_coils)
            if c == useful_coils(i)
            useful_Data_nopy_cp(:,:,i) = Data_nopy_cp(:,:,c);
            useful_csm4ls_sm_cplx(:,i) = csm4ls_sm_cplx(:,c);
            end
        end
    end

    figure, plot(abs(useful_csm4ls_sm_cplx)); 
    
      
    if strcmp(reconType, 'SoS')
    %SOS
    Data_SoS=sqrt(sum(abs(useful_Data_nopy_cp),3).^2);
    recon = Data_SoS;
    end
    if strcmp(reconType, 'wcsm')  
    % Coil Sesnsitivity maps
    Data_py = varargin{1};
    imagval = imag(Data_py);
    realval = real(Data_py);
    absval = abs(Data_py);
    phaseval = angle(Data_py);
    %gausskernel=24; standard
    norm = sqrt(sum(absval,3).^2);

    %compute Coil sensitivity maps
    csm_sm_cplx = complex(imgaussfilt(realval,gausskernel,'FilterDomain','frequency', 'Padding',0 ), imgaussfilt(imagval,gausskernel,'FilterDomain','frequency','Padding',0 ))./imgaussfilt(norm,gausskernel,'FilterDomain','frequency','Padding',0);
    csm_nosm_cplx = Data_py./norm;
    %compute Coil sensitivity lines
    %line_range=330:430;
    line_range = 1:size(csm_nosm_cplx,2);

    csm4ls_sm_cplx = squeeze(sum(csm_sm_cplx(:,line_range,:),2)); % check out range to sum over lines in regions with high SNR, JCWS
    csm4ls_nosm = squeeze(sum(csm_nosm_cplx(:,line_range,:),2));
    %Csm weighted
    for t=1:size(useful_Data_nopy_cp,2)
    Data_csm2(:,t,:) = sum(squeeze(useful_Data_nopy_cp(:,t,:)).*conj(useful_csm4ls_sm_cplx),2)./sqrt(sum(abs(useful_csm4ls_sm_cplx).^2,2));
    end
    Data_csm_abs = abs(Data_csm2);
    recon = Data_csm2;
    end
    
    if strcmp(reconType, 'wtSNR')
    %wtSNR
    tSNRperCoil=squeeze(mean(abs(useful_Data_nopy_cp),2)./std(abs(useful_Data_nopy_cp),[],2)); %tSNR per coil
    for t=1:size(useful_Data_nopy_cp,2)
    Data_wtSNR_num(:,t,:) = squeeze(useful_Data_nopy_cp(:,t,:)).*tSNRperCoil;
    end
    Data_wtSNR=sum(Data_wtSNR_num,3)./sqrt(sum(tSNRperCoil,2).^2);
    Data_wtSNR_abs = abs(Data_wtSNR);
    recon = Data_wtSNR;
    end
    
    if strcmp(reconType, 'wcsmtSNR')
        % Coil Sesnsitivity maps
    Data_py = varargin{1};
    imagval = imag(Data_py);
    realval = real(Data_py);
    absval = abs(Data_py);
    phaseval = angle(Data_py);
    %gausskernel=24; standard
    norm = sqrt(sum(absval,3).^2);

    %compute Coil sensitivity maps
    csm_sm_cplx = complex(imgaussfilt(realval,gausskernel,'FilterDomain','frequency', 'Padding',0 ), imgaussfilt(imagval,gausskernel,'FilterDomain','frequency','Padding',0 ))./imgaussfilt(norm,gausskernel,'FilterDomain','frequency','Padding',0);
    csm_nosm_cplx = Data_py./norm;
    %compute Coil sensitivity lines
    %line_range=330:430;
    line_range = 1:size(csm_nosm_cplx,2);

    csm4ls_sm_cplx = squeeze(sum(csm_sm_cplx(:,line_range,:),2)); % check out range to sum over lines in regions with high SNR, JCWS
    csm4ls_nosm = squeeze(sum(csm_nosm_cplx(:,line_range,:),2));
    %csm+tSNR weighted
    tSNRperCoil=squeeze(mean(abs(useful_Data_nopy_cp),2)./std(abs(useful_Data_nopy_cp),[],2)); %tSNR per coil
    for t=1:size(Data_nopy,2)
    Data_wcsmtSNR(:,t,:) = sum(squeeze(useful_Data_nopy_cp(:,t,:)).*conj(useful_csm4ls_sm_cplx).*tSNRperCoil,2)./sqrt(sum(abs(useful_csm4ls_sm_cplx.*tSNRperCoil).^2,2));
    end
    Data_wcsmtSNR_abs = abs(Data_wcsmtSNR);
    recon = Data_wcsmtSNR;
    end
end

%LS Data
%figure, imagesc(0:TR/1000:dimsSnopy(2)*dimsSnopy(5)*TR/1000,0:0.25:0.25*(dimsSnopy(1)-1), abs(recon)); title('Line-scanning Data'), ylabel('position [mm]'), xlabel('time [s]'), colormap jet;
figure, imagesc(0:TR/1000:size(recon,2)*TR/1000,0:0.25:0.25*(size(recon,1)-1), abs(recon)); title('Line-scanning Data'), ylabel('position [mm]'), xlabel('time [s]'), colormap jet;
colorbar;


 
       
