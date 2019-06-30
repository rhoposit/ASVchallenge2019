% Usage example of cqcc function
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2016 EURECOM, France.
%
% This work is licensed under the Creative Commons
% Attribution-NonCommercial-ShareAlike 4.0 International
% License. To view a copy of this license, visit
% http://creativecommons.org/licenses/by-nc-sa/4.0/
% or send a letter to
% Creative Commons, 444 Castro Street, Suite 900,
% Mountain View, California, 94041, USA.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear

%% ADD CQT TOOLBOX TO THE PATH
addpath('CQT_toolbox_2013');


myFolder = '../../LA_wav_dev'
featFolder = '../../LA_feats_dev/cqcc'
myFiles = dir(fullfile(myFolder,'*.wav'));
%for k = 1:3
for k = 1:length(myFiles)
  baseFileName = myFiles(k).name;
  fullFileName = fullfile(myFolder, baseFileName);
  fprintf(1, 'Now reading %s\n', fullFileName);


  %% INPUT SIGNAL
  [x,fs] = audioread(fullFileName);

  %% PARAMETERS
  B = 96;
  fmax = fs/2;
  fmin = fmax/2^9;
  d = 16;
  cf = 19;
  ZsdD = 'ZsdD';

  %% COMPUTE CQCC FEATURES
  [CQcc, LogP_absCQT, TimeVec, FreqVec, Ures_LogP_absCQT, Ures_FreqVec] = ...
      cqcc(x, fs, B, fmax, fmin, d, cf, ZsdD);


  %CQcc is nCoeff x nFea, since this is DD, the coefs are 20, 20, 20 = 60 total  
  newfile=strrep(baseFileName,'.wav','.h5');
  newfile =fullfile(featFolder,newfile);
  hdf5write(newfile, featFolder, CQcc);
end
