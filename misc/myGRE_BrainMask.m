function emaskfile=myGRE_BrainMask(niifile)
% function emaskfile=myGRE_BrainMask(niifile)
%
% Performs SPM-brain mask, and threshold in outer rim of 3mm

r=3; % rim-radius of 3mm

[path,name,ext]=fileparts(niifile);
mniifile=fullfile(path,['mask_' name ext]);
%%
maskfile=spmBrainMask(niifile);
[path,name,ext]=fileparts(maskfile);
emaskfile=fullfile(path,['e' name ext]);
%%

% load bias-corrected file and threshold
n=nifti(mniifile);
voxelsize=double(n.hdr.pixdim(2:4));
v=n.dat(:,:,:);
t=dip_array(threshold(v));

kern=round(r/mean(voxelsize)); % from mm to voxels

nm=nifti(maskfile);
mask=nm.dat(:,:,:);
%maskorig=mask;
mask=dip_array(closing(opening(mask,kern),kern)); % open and close

% threshold in outer edge
mask_er=mask-erosion(mask,3*kern);
mask(mask_er>0)=t(mask_er>0);
mask=dip_array(closing(opening(mask,kern),kern)); % open and close

nm.dat.fname=emaskfile;
create(nm)
nm.dat(:,:,:)=mask; % write back to file
Mask=mask;