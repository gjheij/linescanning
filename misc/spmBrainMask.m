function varargout=spmBrainMask(niifile,varargin)
% function [maskFile,maskedFile]=spmBrainMask(niifile,[doWriteMaskedFile])
%
% Create a brain mask using SPM segment and threshold.
% doWriteMaskedFile: write masked file with 'masked_' prepended.

doWriteMaskedFile=0;
if nargin==2 & varargin{1}
  doWriteMaskedFile=1;  
end

[path,name,ext]=fileparts(niifile);
if isempty(path)
  path=pwd;
end
% maskFile to create
maskFile=fullfile(path,['mask_' name ext]);
disp(['Creating mask: ' maskFile])

% List of open inputs
% Segment: Data - cfg_files
nrun = 1; % enter the number of runs here
if ~isempty(strfind(spm('ver'),'SPM8'))
  jobfile = {fullfile(fileparts(which('spmBrainMask')),'spmBrainMask_job.m')};
else
  jobfile = {fullfile(fileparts(which('spmBrainMask')),'spmBrainMask_spm12_job.m')};
end
jobs = repmat(jobfile, 1, nrun);
inputs = cell(1, nrun);
for crun = 1:nrun
    inputs{1, crun} = cellstr(niifile); % Segment: Data - cfg_files
    inputs{2, crun} = maskFile; % Imcalc: maskfile
end
spm('defaults', 'PET');
%spm('ver') % should be SPM8
spm_jobman('serial', jobs, '', inputs{:});

%% closing
try
  n=nifti(maskFile);
  mask=n.dat(:,:,:);
  mask=dip_array(closing(mask,3));
  n.dat(:,:,:)=mask;
end

%%
if doWriteMaskedFile
  maskedFile=fullfile(path,['masked_' name ext]);
  disp(['Writing masked: ' maskedFile])
  n=nifti(niifile);
  o=n;
  o.dat.fname=maskedFile;
  create(o);
  o.dat(:,:,:)=ishow(niifile).*ishow(maskFile);

end
%%

if nargout
  varargout{1}=maskFile;
  if doWriteMaskedFile
    varargout{2}=maskedFile;
  end
end