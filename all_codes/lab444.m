clear;clc;

epochs_num=1200; %20 minutes

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
obs_time_S=tabulate(S(:,2));
obs_time_S(1,:)=[];
obs_time_S(:,3)=[];

obs_time_R=tabulate(R(:,2));
obs_time_R(1,:)=[];
obs_time_R(:,3)=[];

obs_time_B=tabulate(B(:,2));
obs_time_B(1,:)=[];
obs_time_B(:,3)=[];

prns_S=tabulate(S(:,1));
prns_S(1,:)=[];
prns_S(:,3)=[];

prns_R=tabulate(R(:,1));
prns_R(1,:)=[];
prns_R(:,3)=[];

prns_B=tabulate(B(:,1));
prns_B(1,:)=[];
prns_B(:,3)=[];

XB=[-1625352.170843933,-3653483.751149267,4953733.869258050];
XB_lla=[51.277,-113.983,1090.833];
GTXR_lla=[51.258643,-114.100492,1127.345];
GTXR=[-1633489.413087291,-3651627.194497138,4952481.599819198];
wgs84 = wgs84Ellipsoid('meter');
GTenu=zeros(1,3);
[GTenu(1),GTenu(2),GTenu(3)]=ecef2enu(GTXR(1),GTXR(2),GTXR(3),XB_lla(1),XB_lla(2),XB_lla(3),wgs84);

%ecef to enu
phi = deg2rad(51.258643);
lambda = deg2rad(-114.100492);
% rotation matrix from ecef to enu
	Rot = [-sin(lambda) cos(lambda) 0 ;
         -cos(lambda)*sin(phi) -sin(lambda)*sin(phi) cos(phi);
         cos(lambda)*cos(phi) cos(phi)*sin(lambda) sin(phi)];
%1 dd results
pose_errors=NaN(epochs_num,3);
sds=NaN(epochs_num,3);
els=NaN(epochs_num,size(prns_R,1));
for i=1:epochs_num
    %find rows of an epoch  size(k,1):number of satellites
    k=find(S(:,2)==obs_time_S(i,1));

    indices_BS=[];
    indices_R=[];
    for j=1:size(k,1)
        p=S(k(j,1),1);
        t=S(k(j,1),2);
        for jj=1:43200
            % do not use satellite  22(9) and 9(3) 18(7)
            if R(jj,1)==p && R(jj,2)==t && R(jj,1)~=22 &&R(jj,1)~=9 &&R(jj,1)~=18
                indices_R(end+1) = jj;
                indices_BS(end+1) = k(j,1);
            end
        end
    end

    %real number of satellites
    num_of_satellites=size(indices_R,2);

    %find max elevation angle
    for j=1:num_of_satellites
        prn=S(indices_BS(1,j),1);%prn of one obs
        prn_idx=int32(find(prns_S(:,1)==prn));
        %get elevation angle
        satPos=[S(indices_BS(1,j),3) S(indices_BS(1,j),4) S(indices_BS(1,j),5)];

        [~,els(i,prn_idx),~] = lookangles(XB_lla,satPos,0);
    end
    el=els(i,:);
    el=el(~isnan(el));

    [~,pivox_index]=max(el,[],2);
    
    %double difference
    XS=zeros(num_of_satellites,3);
    pr_R=zeros(num_of_satellites,1);
    pr_B=zeros(num_of_satellites,1);
    ph_R=zeros(num_of_satellites,1);
    ph_B=zeros(num_of_satellites,1);


    CLIGHT      =299792458.0;         %speed of light (m/s)
    MHZ_TO_HZ     =1000000.0;
    FREQ_GPS_L1   =1575.42*MHZ_TO_HZ;
    lambda=CLIGHT/FREQ_GPS_L1; %L1
    XR=[0;0;0];
    %XR=[-1633459.580317199;-3651644.627857543;4952478.605502964];
    for j=1:num_of_satellites
        XS(j,1)=S(indices_BS(1,j),3);
        XS(j,2)=S(indices_BS(1,j),4);
        XS(j,3)=S(indices_BS(1,j),5);
        pr_R(j,1)=R(indices_R(1,j),3);
        pr_B(j,1)=B(indices_BS(1,j),3);
        ph_R(j,1)=R(indices_R(1,j),4);
        ph_B(j,1)=B(indices_BS(1,j),4);
    end

    [XR,cov_XR,N,cov_N]=dd_code_phase(XR,XB,XS,pr_R,ph_R,pr_B,ph_B,pivox_index,lambda);
    [e,n,u]=ecef2enu(XR(1),XR(2),XR(3),XB_lla(1),XB_lla(2),XB_lla(3),wgs84);
    pose_errors(i,1)=e-GTenu(1);
    pose_errors(i,2)=n-GTenu(2);
    pose_errors(i,3)=u-GTenu(3);
    cov=Rot*cov_XR*Rot';%ecef covariance to enu cov
    sds(i,1)=sqrt(cov(1,1));
    sds(i,2)=sqrt(cov(2,2));
    sds(i,3)=sqrt(cov(3,3));
end
%% 2 plot enu errors 
delta=2;
t=1:delta:epochs_num;
t=t';
subplot(2,2,1)
p=plot(t,pose_errors(1:delta:epochs_num,1),t,sds(1:delta:epochs_num,1),t,-sds(1:delta:epochs_num,1));
p(1).Color='red';
p(2).Color='blue';
p(3).Color='blue';
grid on
xlabel('Time t beginning from 239460(s)')
ylabel('east error(m)')
title('Double Difference East Errors')
legend('east error','std envelope')

subplot(2,2,2)
p=plot(t,pose_errors(1:delta:epochs_num,2),t,sds(1:delta:epochs_num,2),t,-sds(1:delta:epochs_num,2));
p(1).Color='red';
p(2).Color='blue';
p(3).Color='blue';
grid on
xlabel('Time t beginning from 239460(s)')
ylabel('north error(m)')
title('Double Difference North Errors')
legend('north error','std envelope')

subplot(2,2,3)
p=plot(t,pose_errors(1:delta:epochs_num,3),t,sds(1:delta:epochs_num,3),t,-sds(1:delta:epochs_num,3));
p(1).Color='red';
p(2).Color='blue';
p(3).Color='blue';
grid on
xlabel('Time t beginning from 239460(s)')
ylabel('up error(m)')
title('Double Difference Up Errors')
legend('up error','std envelope')

%% 3 rmse
sum=0;
for i=1:epochs_num
    sum=sum+pose_errors(i,1)*pose_errors(i,1);
end
rmse_e=sqrt(sum/epochs_num);

sum=0;
for i=1:epochs_num
    sum=sum+pose_errors(i,2)*pose_errors(i,2);
end
rmse_n=sqrt(sum/epochs_num);

sum=0;
for i=1:epochs_num
    sum=sum+pose_errors(i,3)*pose_errors(i,3);
end
rmse_u=sqrt(sum/epochs_num);