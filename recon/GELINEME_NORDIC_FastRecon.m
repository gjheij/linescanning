%% Reconstruction for multi echo line scanning (GELINEME)
% outputs:recon = complex recosntruction with tSNR/channel and csm
%         varargout = number of removed components when nordic in on, for every
%         channel and every echo
% inputs: csm4ls_sm_cplx = coil sensitivities, base on data with phase encoding on and
%         rest slabs, from coil_sens_MELS.m function 
%         data_lab = data in lab format
%         filelocation = path with the location of your data
%         Nc = total number of channels
%         Nechos = number of echoes
%         TR = repetition time in ms
%         Nechos = number of echos
%         sp_res = spatial resolution in mm along teh line
%         flyback = can be 'on' and 'off' and depends on the scanner
%         settings (if you have monopolar or bipolar readout. Normally off
%         NORDIC = can be 'on' and 'off', depending if you want to perorm
%         nordic denoising
%         varagin = is a cell array containing at the first place if NORDIC
%         neeeds to be applied 'before' fourier transform or 'after' it. 
%         second argument of the cell array is the NORDIC treshold used in
%         case NORDIC is on (depending on which kind of thresholding we are
%         using. If nordic_ev function is called, no threhold is needed,
%         since the elbow of the scree plot is used to select the
%         threshold).

function [recon, varargout] = GELINEME_NORDIC_FastRecon(csm4ls_sm_cplx, nopy_lab, filelocation, Nc, Nechos, TR, sp_res, flyback, NORDIC, varargin)

%Nechos = size(Data_py,length(size(Data_py)));
Snopy=MRecon([filelocation nopy_lab]);
Snopy.Parameter.Parameter2Read.typ = 1;
Snopy.ReadData;
Snopy.PDACorrection;
Snopy.RandomPhaseCorrection;
Snopy.RemoveOversampling;
Snopy.DcOffsetCorrection;
Snopy.MeasPhaseCorrection;
Snopy.SortData;
Ndyn = size(Snopy.Data,length(size(Snopy.Data))-2);
Snopy.PartialFourier;

D = Snopy.Data;
%compensate for flyback off in k-space
if strcmp(flyback,'off')
D_norev = D;
D_flip = flip(D,1);

D1 = zeros(size(D));
for ch = 1:32
for dyn =1:Ndyn
    for e = 1:Nechos
        if rem(e,2) == 0
        D1(:,:,1,ch,dyn,1,e) = D_flip(:,:,1,ch,dyn,1,e);
        else
        D1(:,:,1,ch,dyn,1,e) = D_norev(:,:,1,ch,dyn,1,e);
        end
    end
end
end
Snopy.Data=D1;

elseif strcmp(flyback,'on')
Snopy.Data = D;
end

Snopy.Data = squeeze(Snopy.Data);

dimsSnopy=size(Snopy.Data);
size(Snopy.Data)

% more dynamics
Snopy.Data = flip(Snopy.Data,2); %each dynamic has time reversed!!! You correct fot that here
Data_nopy_norev = Snopy.Data;
Data_nopy_norev=reshape(permute(Data_nopy_norev, [1, 2, 4, 3, 5]), dimsSnopy(1),dimsSnopy(2)*dimsSnopy(4),dimsSnopy(3), dimsSnopy(5));

%% NORDIC before FT
Data_nordic = zeros(size(Data_nopy_norev));

if strcmp(NORDIC,'on') && strcmp(varargin{1}{1,1},'before')
    disp ('NORDIC before FT')
    n_thresh = varargin{1}{1,2};
    if n_thresh>0

        % multiply by 100 if proportion was given
        if n_thresh<1
            n_thresh = n_thresh*100
        end

        % use single quotes!
        X = ['Removing ', num2str(n_thresh), '% of components '];
        disp(X)
    else
        disp("Using scree-plot to remove components")
    end

    rem_comp = zeros(Nc*Nechos,1);
    ii = 1;
    for ch =1:Nc
        for e = 1:Nechos

            % use nordic_psr to remove specific percentage of components; if n_thresh==0, use elbow criteria
            if n_thresh>0
                % use single quotes!
                % X = ['Removing ', num2str(n_thresh), '% of components '];
                % disp(X)
                [Data_nordic(:,:,ch,e), rem_comp(ii)] = nordic_psr(n_thresh,Data_nopy_norev(:,:,ch,e));
            else
                % disp("Using scree-plot to remove components")
                [Data_nordic(:,:,ch,e), rem_comp(ii)] = nordic_ev(Data_nopy_norev(:,:,ch,e));
                % X = ['Removed ', num2str(rem_comp(ii),'%4.2f') '% of components'];
            end
            ii = ii + 1;         
        end
    end
    varargout{1} = rem_comp;
    Data_nopy_norev = Data_nordic;
    
elseif strcmp(NORDIC,'off')
    disp ('NORDIC off')
    Data_nopy_norev = Data_nopy_norev;
end

%% 
Snopy.Data = Data_nopy_norev;
Snopy.K2IM; % fft in readout direction
Data_nopy_norev = Snopy.Data;
 

%% NORDIC after FT but before coil combination 
Data_nopy_norev_nordic = zeros(size(Data_nopy_norev));
if strcmp(NORDIC,'on') && strcmp(varargin{1}{1,1},'after')
    disp ('NORDIC after FT')
    for ch =1:Nc
        for e = 1:Nechos
            Data_nopy_norev_nordic(:,:,ch,e) = nordic_psr(varargin{1}{1,2},Data_nopy_norev(:,:,ch,e));
        end
    end
    Data_nopy_norev = Data_nopy_norev_nordic;
elseif strcmp(NORDIC,'off')
    Data_nopy_norev = Data_nopy_norev;
    disp ('NORDIC off')
end

%% Reconstuction: coil combination with tSNR and csm weighting

%csm+tSNR weighted
tSNRperCoil=squeeze(mean(abs(Data_nopy_norev),2)./std(abs(Data_nopy_norev),[],2)); %tSNR per coil
for t=1:size(Data_nopy_norev,2)
Data_wcsmtSNR_norev(:,t,:,:) = sum(squeeze(Data_nopy_norev(:,t,:,:)).*conj(csm4ls_sm_cplx).*tSNRperCoil,2)./sqrt(sum(abs(csm4ls_sm_cplx.*tSNRperCoil).^2,2));
end

Data_wcsmtSNR = Data_wcsmtSNR_norev;

%% tSNR evaluation
tSNR_wcsmtSNR=squeeze(mean(abs(Data_wcsmtSNR),2)./std(abs(Data_wcsmtSNR),[],2));
bs9=mean(tSNR_wcsmtSNR(1:10,:));

 %%  Figures more dyn
%tSNR
figure
for echo = 1:Nechos 
plot(1:size(Data_nopy_norev,1), tSNR_wcsmtSNR(:,echo)-bs9);
xlabel('voxels');
hold on;
end
title('tSNR_{wcsmtSNR} for every echo');

%LS Data

for echo =1:Nechos
figure, imagesc(0:TR/1000:dimsSnopy(2)*dimsSnopy(5)*TR/1000,0:0.25:0.25*(dimsSnopy(1)-1), abs(Data_wcsmtSNR(:,:,echo))); title('Human V1, wcsmtSNR Data'), ylabel('position (mm)'), xlabel('time [s]'), colormap parula;
caxis([0 35]);
colorbar;
end

recon = Data_wcsmtSNR;
 
