%% Coil Sesnsitivity maps for multi-echo line-scanning (MELS)
%         Data_py = Data per channel of the LSD image for every channel
%         (use basic_recon.m)
%         gausskernel = FWHM of the gaussian for the smoothing kernel
%         (standard is 24)
function cs = coil_sens_MELS(Data_py, gausskernel)

%compute Coil sensitivity maps
e = 1;
Data_csm = squeeze(Data_py(:,:,:,e));
norm = sqrt(sum(abs(Data_csm),3).^2);
csm_sm_cplx = complex(imgaussfilt(real(Data_csm),gausskernel,'FilterDomain','frequency', 'Padding',0 ), imgaussfilt(imag(Data_csm),gausskernel,'FilterDomain','frequency','Padding',0 ))./imgaussfilt(norm,gausskernel,'FilterDomain','frequency','Padding',0);


%compute Coil sensitivity lines
%line_range=330:390; 
line_range = 1:size(csm_sm_cplx,2);

csm4ls_sm_cplx = squeeze(sum(csm_sm_cplx(:,line_range,:,:),2)); % check out range to sum over lines in regions with high SNR, JCWS


figure, montage(abs(csm_sm_cplx(:,:,:)),'DisplayRange',[]);
title('coil sensitivities');
figure, montage(abs(csm4ls_sm_cplx),'DisplayRange',[]);

cs = csm4ls_sm_cplx;
end