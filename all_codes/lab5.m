format long;
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
obs_time=tabulate(S(:,2));
obs_time(1,:)=[];
obs_time(:,3)=[];

prns=tabulate(S(:,1));
prns(1,:)=[];
prns(:,3)=[];

XB=[-1625352.170843933,-3653483.751149267,4953733.869258050];
XB_lla=[51.277,-113.983,1090.833];
GTXR_lla=[51.258643,-114.100492,1127.345];
GTXR=[-1633489.413087291,-3651627.194497138,4952481.599819198];
wgs84 = wgs84Ellipsoid('meter');
GTenu=zeros(1,3);
[GTenu(1),GTenu(2),GTenu(3)]=ecef2enu(GTXR(1),GTXR(2),GTXR(3),XB_lla(1),XB_lla(2),XB_lla(3),wgs84);


% %0 iono correction
% ionB=NaN(epochs_num,size(prns,1));
% ionR=NaN(epochs_num,size(prns,1));
% for i=1:epochs_num
%     %find rows of an epoch  size(k,1):number of satellites
%     k=find(S(:,2)==obs_time(i,1));
% 
%     %find max elevation angle
%     for j=1:size(k,1)
%         prn=S(k(j,1),1);%prn of one obs
%         prn_idx=int32(find(prns(:,1)==prn));
%         %get elevation angle
%         satPos=[S(k(j,1),3) S(k(j,1),4) S(k(j,1),5)];
% 
%         sow=S(k(j,1),2);
%         [a1,e1,~] = lookangles(XB_lla,satPos,0);
%         ionB(i,prn_idx)=iono_corr(sow,XB_lla(1),XB_lla(2),a1,e1);
% 
%         [a2,e2,~] = lookangles(GTXR_lla,satPos,0);
%         ionR(i,prn_idx)=iono_corr(sow,GTXR_lla(1),GTXR_lla(2),a2,e2);
%     end
% end

%% 1 dd

els=NaN(epochs_num,size(prns,1));
ratios=NaN(epochs_num,1);
enus=NaN(epochs_num,3);
stds=NaN(epochs_num,3);
for i=1:epochs_num
    %find rows of an epoch  size(k,1):number of satellites
    k=find(S(:,2)==obs_time(i,1));

    %find max elevation angle
    for j=1:size(k,1)
        prn=S(k(j,1),1);%prn of one obs
        prn_idx=int32(find(prns(:,1)==prn));
        %get elevation angle
        satPos=[S(k(j,1),3) S(k(j,1),4) S(k(j,1),5)];

        [~,els(i,prn_idx),~] = lookangles(XB_lla,satPos,0);
    end
    el=els(i,:);
    el=el(~isnan(el));

    [~,pivox_index]=max(el,[],2);
    
    %double difference
    XS=zeros(size(k,1),3);
    pr_R=zeros(size(k,1),1);
    pr_B=zeros(size(k,1),1);
    ph_R=zeros(size(k,1),1);
    ph_B=zeros(size(k,1),1);
    ion_R=ionR(i,:)';
    ion_R=ion_R(~isnan(ion_R));
    ion_B=ionB(i,:)';
    ion_B=ion_B(~isnan(ion_B));

    CLIGHT      =299792458.0;         %speed of light (m/s)
    MHZ_TO_HZ     =1000000.0;
    FREQ_GPS_L1   =1575.42*MHZ_TO_HZ;
    lambda=CLIGHT/FREQ_GPS_L1; %L1
    %XR=[0;0;0];
    XR=[-1633459.580317199;-3651644.627857543;4952478.605502964];
    for j=1:size(k,1)
        XS(j,1)=S(k(j,1),3);
        XS(j,2)=S(k(j,1),4);
        XS(j,3)=S(k(j,1),5);
        pr_R(j,1)=R(k(j,1),3);
        pr_B(j,1)=B(k(j,1),3);
        ph_R(j,1)=R(k(j,1),4);
        ph_B(j,1)=B(k(j,1),4);
    end

    [XR,cov_XR,N,cov_N]=dd_code_phase(XR,XB,XS,pr_R,ph_R,pr_B,ph_B,pivox_index,lambda,ion_R,ion_B);
    if any(any(isinf(XR)))||any(any(isinf(cov_XR)))...
            ||any(any(isinf(N)))||any(any(isinf(cov_N)))
        continue;
    end
    if any(any(isnan(XR)))||any(any(isnan(cov_XR)))...
            ||any(any(isnan(N)))||any(any(isnan(cov_N)))
        continue;
    end
    stds(i,1)=sqrt(cov_XR(1,1));
    stds(i,2)=sqrt(cov_XR(2,2));
    stds(i,3)=sqrt(cov_XR(3,3));
    if abs(stds(i,1))>100||abs(stds(i,2))>100||abs(stds(i,3))>100
        continue;
    end
    [enus(i,1),enus(i,2),enus(i,3)]=ecef2enu(XR(1),XR(2),XR(3),XB_lla(1),XB_lla(2),XB_lla(3),wgs84);
    
    [NN,r]=mlambda(cov_N,N,4);
    
    ratios(i)=r(2)/r(1);%>2
    % [U] = chol(cov_N); 
    % cov_N = U'*U; 
    % [afix, sqnorm, Qahat, Z, D, L] = lambda_routine2(N,cov_N); %using LAMBDA routines
    % acheck = afix(:,1);
    % Ps = prod(2*normcdf(0.5./sqrt(D))-1);
    % 
    % P0 = 0.001;
    % if (1-Ps > P0)
    % mu = ratioinv(P0,1-Ps,length(acheck));
    % else
    % mu = 1;
    % end
    % ratio = sqnorm(1)/sqnorm(2);
    % if abs(stds(i,1))>100||abs(stds(i,2))>100||abs(stds(i,3))>100
    %     enus(i,:)=NaN(1,3);
    %     ratios(i)=NaN;
    %     stds(i,:)=NaN(1,3);
    % end
    
    %fix ambiguity
    
end
