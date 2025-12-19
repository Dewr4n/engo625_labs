clear;clc;

epochs_num=3600;
maskAngle=5;%degree

recPos=[51.258643 -114.100492 1127.345];

%read base station, 6 columns
%fid = fopen('C:\Users\yandi\Downloads\DataSet\BaseL1L2.obs');
%B = fread(fid,[6 Inf],'double')';

%read rover station, 6 columns
fid = fopen('/MATLAB Drive/RemoteL1L2.obs');
R= fread(fid,[6 Inf],'double')';

%read satellite, 8 columns
fid = fopen('/MATLAB Drive/Satellites.sat');
S = fread(fid,[8 Inf],'double')';

%get observation time and prn
obs_time=tabulate(S(:,2));
obs_time(1,:)=[];
obs_time(:,3)=[];

prns=tabulate(S(:,1));
prns(1,:)=[];
prns(:,3)=[];

%set colors
color_types=int32(size(prns,1));
colors=zeros(color_types,3);
colors(:,1)=rand(color_types,1);
colors(:,2)=rand(color_types,1);
colors(:,3)=rand(color_types,1);

LegendsStrings = cell(size(prns,1),1); % Initialize array  with legends
% 1 spp
poses=NaN(epochs_num,4);
residuals=NaN(epochs_num,size(prns,1));%pr residual
stds=NaN(epochs_num,3);%enu td
els=NaN(epochs_num,size(prns,1));
%ecef to enu
phi = deg2rad(51.258643);
lambda = deg2rad(-114.100492);
% rotation matrix from ecef to enu
	Rot = [-sin(lambda) cos(lambda) 0 0;
         -cos(lambda)*sin(phi) -sin(lambda)*sin(phi) cos(phi) 0;
         cos(lambda)*cos(phi) cos(phi)*sin(lambda) sin(phi) 0
         0 0 0 1];
     
for i=1:epochs_num
    %find rows of an epoch  size(k,1):number of satellites
    k=find(S(:,2)==obs_time(i,1));
    sat_poses=zeros(3,size(k,1));
    ranges=zeros(1,size(k,1));
    
    for j=1:size(k,1)
        sat_poses(1,j)=S(k(j,1),3);
        sat_poses(2,j)=S(k(j,1),4);
        sat_poses(3,j)=S(k(j,1),5);
        ranges(1,j)=R(k(j,1),3);
    end
    [pose,res,cov]=SPP(sat_poses,ranges);
    poses(i,:)=pose';
    cov=Rot*cov*Rot';%ecef covariance to enu cov
    stds(i,1)=sqrt(cov(1,1));
    stds(i,2)=sqrt(cov(2,2));
    stds(i,3)=sqrt(cov(3,3));
    %PR residuals
    for j=1:size(k,1)
        if abs(res(j))>100  ||abs(res(j))==0
            continue
        end
        prn=S(k(j,1),1);%prn of one obs
        prn_idx=int32(find(prns(:,1)==prn));
        residuals(i,prn_idx)=res(j);

        %get elevation angle
        satPos=[S(k(j,1),3) S(k(j,1),4) S(k(j,1),5)];

        [~,els(i,prn_idx),~] = lookangles(recPos,satPos,maskAngle);
    end
   
end

%% 3 draw pr residuals

%4 pr with regard to elevation angle

plot(els(1:2:epochs_num,:),residuals(1:2:epochs_num,:));
grid on
xlabel('Elevation Angle(degree)')
ylabel('PR residual(m)')
title('PR Residuals regarding to Elevation Angle')
legend('7','8','9',...
    '11','15','17',...
    '18','19','22',...
    '24','26','27',...
    '29','NumColumns',3);