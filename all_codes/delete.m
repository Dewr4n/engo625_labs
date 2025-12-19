clear;clc;

epochs_num=1;

%read base station, 6 columns
fid = fopen('/MATLAB Drive/BaseL1L2.obs');
B = fread(fid,[6 Inf],'double')';

%read rover station, 6 columns
fid = fopen('/MATLAB Drive/RemoteL1L2.obs');
R= fread(fid,[6 Inf],'double')';

%read satellite, 8 columns
fid = fopen('/MATLAB Drive/Satellites.sat');
S = fread(fid,[8 Inf],'double')';

%get observation time and prn
%obs_time_S AND obs_time_B SAME
%prns_S AND prns_B SAME
