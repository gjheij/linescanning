%%Recon with complex values for every channel separately
%if you want to have same geometry as ParRec you have to rot90 and fliplr
%(but you don't have to for the purpose of line-scanning recon)
function output = basic_recon(data_lab,filelocation,Nc)

size_set = 720;
S=MRecon(fullfile(filelocation, data_lab));
S.Parameter.Parameter2Read.typ = 1;
S.ReadData;
S.PDACorrection;
S.RandomPhaseCorrection;
S.RemoveOversampling; 
S.DcOffsetCorrection;
S.MeasPhaseCorrection;
S.SortData;
S.PartialFourier; %kind of correrction
S.GridData;

%manual zero filling, since gyrotools function is not working

D = S.Data;
add = (size_set - size(S.Data,2))/2;
D = padarray(D,[0,add],0,'pre');
D = padarray(D,[0,add],0,'post');

S.Data=D;

S.K2I; %fft in readout and phase direction together
%S.Data=fftshift(S.Data,2); %you have to comment when usign version
%3.0.557, 4.2.2 and 4.3.1
%and uncomment when using 3.0.541
S.GridderNormalization;

Data_py= squeeze(squeeze(S.Data(:,:,1,:,:,1,:))); %complex values

% figure,montage(abs(Data_py(:,:,:,1,1)),'Indices', 1:Nc,'DisplayRange',[]); title('all channels, 1 dyn, 1st echo(in case of more)');
% figure,montage(angle(Data_py(:,:,:,1,1)),'Indices', 1:Nc,'DisplayRange',[]); title('all channels, 1 dyn, 1st echo(in case of more)');

% Datapy_SoS=sqrt(sum(abs(Data_py),3).^2);
% figure, imshow(Datapy_SoS,[]);
% title('My Recon SoS');

imagval = imag(Data_py);
realval = real(Data_py);
absval = abs(Data_py);
phaseval = angle(Data_py);

output = Data_py;
