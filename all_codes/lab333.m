clear;clc;

epochs_num=3600;

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

%set colors
color_types=int32(size(prns_R,1));
colors=zeros(color_types,3);
colors(:,1)=rand(color_types,1);
colors(:,2)=rand(color_types,1);
colors(:,3)=rand(color_types,1);

LegendsStrings = cell(size(prns_R,1),1); % Initialize array  with legends

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
	Rot = [-sin(lambda) cos(lambda) 0 0;
         -cos(lambda)*sin(phi) -sin(lambda)*sin(phi) cos(phi) 0;
         cos(lambda)*cos(phi) cos(phi)*sin(lambda) sin(phi) 0
         0 0 0 1];

pose_errors=NaN(epochs_num,3);
residuals=NaN(epochs_num,size(prns_R,1));%pr residual
sds=NaN(epochs_num,3);
els=NaN(epochs_num,size(prns_R,1));
%% 1 sd results
for i=1:epochs_num
    %INDEX: find rows of an epoch  size(k,1):number of satellites
    k=find(S(:,2)==obs_time_S(i,1));%ith time

    indices_BS=[];
    indices_R=[];
    for j=1:size(k,1)
        p=S(k(j,1),1);
        t=S(k(j,1),2);
        for jj=1:43200
            if R(jj,1)==p && R(jj,2)==t
                indices_R(end+1) = jj;
                indices_BS(end+1) = k(j,1);
            end
        end
    end

    %real number of satellites
    num_of_satellites=size(indices_R,2);

    %single difference
    XS=zeros(num_of_satellites,3);
    pr_R=zeros(num_of_satellites,1);
    pr_B=zeros(num_of_satellites,1);

    
    XR=[0;0;0;0];
    %XR=[-1633459.580317199;-3651644.627857543;4952478.605502964;0];
    for j=1:num_of_satellites
        XS(j,1)=S(indices_BS(1,j),3);
        XS(j,2)=S(indices_BS(1,j),4);
        XS(j,3)=S(indices_BS(1,j),5);
        pr_R(j,1)=R(indices_R(1,j),3);
        pr_B(j,1)=B(indices_BS(1,j),3);
    end
    [XR,cov_XR,res]=sd_code(XR,XB,XS,pr_R,pr_B);
    [e,n,u]=ecef2enu(XR(1),XR(2),XR(3),XB_lla(1),XB_lla(2),XB_lla(3),wgs84);
    pose_errors(i,1)=e-GTenu(1);
    pose_errors(i,2)=n-GTenu(2);
    pose_errors(i,3)=u-GTenu(3);
    cov=Rot*cov_XR*Rot';%ecef covariance to enu cov
    sds(i,1)=sqrt(cov(1,1));
    sds(i,2)=sqrt(cov(2,2));
    sds(i,3)=sqrt(cov(3,3));
    %residuals
    for j=1:num_of_satellites
        p=S(indices_BS(1,j),1);
        prn_idx=int32(find(prns_S(:,1)==p));
        residuals(i,prn_idx)=res(j);

        %get elevation angle
        satPos=[S(indices_BS(1,j),3) S(indices_BS(1,j),4) S(indices_BS(1,j),5)];

        [~,els(i,prn_idx),~] = lookangles(GTXR_lla,satPos,5);
    end

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
title('PR single difference east errors')
legend('east error','std envelope')

subplot(2,2,2)
p=plot(t,pose_errors(1:delta:epochs_num,2),t,sds(1:delta:epochs_num,2),t,-sds(1:delta:epochs_num,2));
p(1).Color='red';
p(2).Color='blue';
p(3).Color='blue';
grid on
xlabel('Time t beginning from 239460(s)')
ylabel('north error(m)')
title('PR single difference north errors')
legend('north error','std envelope')

subplot(2,2,3)
p=plot(t,pose_errors(1:delta:epochs_num,3),t,sds(1:delta:epochs_num,3),t,-sds(1:delta:epochs_num,3));
p(1).Color='red';
p(2).Color='blue';
p(3).Color='blue';
grid on
xlabel('Time t beginning from 239460(s)')
ylabel('up error(m)')
title('PR single difference up errors')
legend('up error','std envelope')
%% 3 plot res
% delta=2;
% t=1:delta:epochs_num;
% plot(t,residuals(1:delta:epochs_num,1),t,residuals(1:delta:epochs_num,2),...
%     t,residuals(1:delta:epochs_num,3),t,residuals(1:delta:epochs_num,4),...
%     t,residuals(1:delta:epochs_num,5),t,residuals(1:delta:epochs_num,6),...
%     t,residuals(1:delta:epochs_num,7),t,residuals(1:delta:epochs_num,8),...
%     t,residuals(1:delta:epochs_num,9),t,residuals(1:delta:epochs_num,10),...
%     t,residuals(1:delta:epochs_num,11),t,residuals(1:delta:epochs_num,12),...
%     t,residuals(1:delta:epochs_num,13));
% 
% grid on
% xlabel('Time t beginning from 239460(s)')
% ylabel('residual(m)')
% title('Time Series of single difference Residuals')
% legend('7','8','9',...
%     '11','15','17',...
%     '18','19','22',...
%     '24','26','27',...
%     '29')
%% 4 with regard to elevation angle
% delta=2;
% plot(els(1:delta:epochs_num,:),residuals(1:delta:epochs_num,:));
% grid on
% xlabel('Elevation Angle(degree)')
% ylabel('single difference PR residual(m)')
% title('Single Difference PR Residuals regarding to Elevation Angle')
% legend('7','8','9',...
%     '11','15','17',...
%     '18','19','22',...
%     '24','26','27',...
%     '29','NumColumns',3);
%% 5 rmse
% sum=0;
% for i=1:epochs_num
%     sum=sum+pose_errors(i,1)*pose_errors(i,1);
% end
% rmse_e=sqrt(sum/epochs_num);
% 
% sum=0;
% for i=1:epochs_num
%     sum=sum+pose_errors(i,2)*pose_errors(i,2);
% end
% rmse_n=sqrt(sum/epochs_num);
% 
% sum=0;
% for i=1:epochs_num
%     sum=sum+pose_errors(i,3)*pose_errors(i,3);
% end
% rmse_u=sqrt(sum/epochs_num);