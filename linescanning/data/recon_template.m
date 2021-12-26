%% paths
addpath(getenv('MRRECON')) %working without patch or with seline patch
addpath(getenv('MATLAB_DIR'))
addpath('/data1/projects/MicroFunc/common');
addpath(genpath(fullfile(getenv('PATH_HOME'), 'programs', 'luisa')))
disp("Starting recon")
filelocation = 'INPUT_DIR';


% py_PAR   = 'RECON_PAR'; % sub-xxx_desc-recon.PAR
py_lab   = 'RECON_LAB';  % sub-xxx_desc-recon.LAB
nopy_lab = 'TASK_LAB'; % sub-xxx_task-2R_bold.LAB
nordic = 'DO_NORDIC';

Nc = 32; % nr of channels
TR = 105; %TR in ms
sp_res = 0.25; %line resolution [mm]
gausskernel = 24;

Data_pc       = basic_recon(py_lab,filelocation, Nc);
nord_tresh    = 9; %not used now, since you are selecting the NORDIC threshold based on the scree plot 
varargin_par  = {Data_pc, nord_tresh};
Data_wcsmtSNR = LS_recon_1way(nopy_lab, filelocation, Nc, 'all', TR, gausskernel, 'no', 'wcsmtSNR', nordic, varargin_par);

save('OUTPUT_MAT','Data_wcsmtSNR');
disp("Done")

%
% %% %% Data_py
% %[lsData_py, info_py]=loadParRec([filelocation py_PAR]); %image domain
%
% %if you have 2 dyn (one normal and one only noise) use the following 2
% %rows. otherwise the previous one
% [lsData_py_2dyn, info_py]=loadParRec(fullfile(filelocation, py_PAR)); %image domain
% lsData_py = lsData_py_2dyn(:,:,:,:,1);
%
% S=MRecon([filelocation py_lab]);
% S.Parameter.Parameter2Read.typ = 1;
% S.ReadData;
% S.PDACorrection;
% S.RandomPhaseCorrection;
% S.RemoveOversampling;
% S.DcOffsetCorrection;
% S.MeasPhaseCorrection;
% S.SortData;
% S.PartialFourier; %kind of correrction
% S.GridData;
%
% %manual zero filling, since gyrotools function is not working
% ZP1=zeros(size(S.Data,1),(size(lsData_py,2)-size(S.Data,2))./2);
%
% for i = 1:Nc
% C(:,:,1,i,1)=[ZP1,S.Data(:,:,1,i,1)];
% end
% size(C);
%
% ZP2=zeros(size(S.Data,1),size(lsData_py,2)-size(C,2));
%
% for i = 1:Nc
% D(:,:,1,i,1)=[C(:,:,1,i,1),ZP2];
% end
% size(D);
%
% S.Data=D;
%
% S.K2IM;
% S.K2IP;
% %S.K2I; %fft in readout and phase direction together
% S.Data=fftshift(S.Data,2);
% S.GridderNormalization;
%
% Data_py= squeeze(S.Data(:,:,:,:,1)); %complex values
%
% % figure,montage(abs(Data_py),'Indices', 1:Nc,'DisplayRange',[]);
%
% Datapy_SoS=sqrt(sum(abs(Data_py),3).^2);
% %Datapy_SoS=sqrt(sum(abs(Data_py(:,:,27:28)),3).^2);
% % figure, imshow(Datapy_SoS,[]);
% % title('My Recon');
%
% imagval = imag(Data_py);
% realval = real(Data_py);
% absval = abs(Data_py);
% phaseval = angle(Data_py);
% %figure,montage(log(absval),'Indices', 1:32,'DisplayRange',[]);
%
% %%  Data_nopy
% Snopy=MRecon([filelocation nopy_lab]);
% Snopy.Parameter.Parameter2Read.typ = 1;
% Snopy.ReadData;
% Snopy.PDACorrection;
% Snopy.RandomPhaseCorrection;
% Snopy.RemoveOversampling;
% Snopy.DcOffsetCorrection;
% Snopy.MeasPhaseCorrection;
% Snopy.SortData;
% Snopy.PartialFourier;
%
% Snopy.K2IM; % fft in readout direction
%
% %Snopy.RemoveOversampling;
% dimsSnopy=size(Snopy.Data);
% size(Snopy.Data)
% % % 1 dynamic
% % Data_nopy=squeeze(abs(Snopy.Data));
% % Data_nopy_cp=squeeze(Snopy.Data);
%
% % more dynamics
% Snopy.Data = flip(Snopy.Data,2); %each dynamic has time reversed!!! You correct fot that here
% Data_nopy=reshape(permute(abs(squeeze(Snopy.Data)), [1, 2, 4, 3]), dimsSnopy(1),dimsSnopy(2)*dimsSnopy(5),dimsSnopy(4));
% Data_nopy_cp=reshape(permute(squeeze(Snopy.Data), [1, 2, 4, 3]), dimsSnopy(1),dimsSnopy(2)*dimsSnopy(5),dimsSnopy(4));
%
%
% %% Coil Sesnsitivity maps
% gausskernel=24;
% norm = sqrt(sum(absval,3).^2);
%
% %compute Coil sensitivity maps
% csm_sm_cplx = complex(imgaussfilt(realval,gausskernel,'FilterDomain','frequency', 'Padding',0 ), imgaussfilt(imagval,gausskernel,'FilterDomain','frequency','Padding',0 ))./imgaussfilt(norm,gausskernel,'FilterDomain','frequency','Padding',0);
% csm_nosm_cplx = Data_py./norm;
% %compute Coil sensitivity lines
% line_range=250:450;
% %line_range = 1:size(csm_nosm_cplx,2);
%
% csm4ls_sm_cplx = squeeze(sum(csm_sm_cplx(:,line_range,:),2)); % check out range to sum over lines in regions with high SNR, JCWS
% csm4ls_nosm = squeeze(sum(csm_nosm_cplx(:,line_range,:),2));
%
% %% Reconstuction: different ways
%
% %SOS
% Data_SoS=sqrt(sum(abs(Data_nopy_cp),3).^2);
%
% %Csm weighted
% for t=1:size(Data_nopy,2)
% Data_csm2(:,t,:) = sum(squeeze(Data_nopy_cp(:,t,:)).*conj(csm4ls_sm_cplx),2)./sqrt(sum(abs(csm4ls_sm_cplx).^2,2));
% end
% Data_csm_abs = abs(Data_csm2);
%
% %wtSNR
% tSNRperCoil=squeeze(mean(abs(Data_nopy_cp),2)./std(abs(Data_nopy_cp),[],2)); %tSNR per coil
% for t=1:size(Data_nopy_cp,2)
% Data_wtSNR_num(:,t,:) = squeeze(Data_nopy_cp(:,t,:)).*tSNRperCoil;
% end
% Data_wtSNR=sum(Data_wtSNR_num,3)./sqrt(sum(tSNRperCoil,2).^2);
% Data_wtSNR_abs = abs(Data_wtSNR);
%
%
% %csm+tSNR weighted
% for t=1:size(Data_nopy,2)
% Data_wcsmtSNR(:,t,:) = sum(squeeze(Data_nopy_cp(:,t,:)).*conj(csm4ls_sm_cplx).*tSNRperCoil,2)./sqrt(sum(abs(csm4ls_sm_cplx.*tSNRperCoil).^2,2));
% end
% Data_wcsmtSNR_abs = abs(Data_wcsmtSNR);
%
% %% tSNR evaluation
% tSNR_SoS=squeeze(mean(Data_SoS,2)./std(Data_SoS,[],2));
% bs3=mean(tSNR_SoS(1:10));
%
% tSNR_csm=squeeze(mean(abs(Data_csm2),2)./std(abs(Data_csm2),[],2));
% bs5=mean(tSNR_csm(1:10));
%
% tSNR_wtSNR=squeeze(mean(abs(Data_wtSNR),2)./std(abs(Data_wtSNR),[],2));
% bs7=mean(tSNR_wtSNR(1:10));
%
% tSNR_wcsmtSNR=squeeze(mean(abs(Data_wcsmtSNR),2)./std(abs(Data_wcsmtSNR),[],2));
% bs9=mean(tSNR_wcsmtSNR(1:10));

% save('OUTPUT_MAT', 'Data_csm_abs', 'Data_SoS', 'Data_wtSNR_abs', 'Data_wcsmtSNR_abs', 'Data_wcsmtSNR' );
