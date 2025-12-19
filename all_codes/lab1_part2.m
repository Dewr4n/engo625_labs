clear;clc;

satIDs=[7;8;9;11;15;17;18;19;22;24;26;27;28];

epochs_num=3600;
maskAngle=0;

recPos=[51.258643 -114.100492 1127.345];

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

az = zeros(epochs_num,size(prns,1));
el = zeros(epochs_num,size(prns,1));
vis = false(epochs_num,size(prns,1));
%3 calculate dop
dops=zeros(epochs_num,4);
%sp = skyplot([],[],MaskElevation=maskAngle);
%title('Skyplot of GPS satellites(azimuth and elevation)');
% %read epochs
for i=1:epochs_num
    %find rows of an epoch
    k=find(S(:,2)==obs_time(i,1));

    for j=1:size(k,1)
        prn=S(k(j,1),1);%prn of one obs
        prn_idx=int32(find(prns(:,1)==prn));

        satPos=[S(k(j,1),3) S(k(j,1),4) S(k(j,1),5)];

        [az(i,prn_idx),el(i,prn_idx),vis(i,prn_idx)] = lookangles(recPos,satPos,maskAngle);
        az(az==0)=nan;
        el(el==0)=nan;
        if vis(i,prn_idx)==0
            az(i,prn_idx)=nan;
            el(i,prn_idx)=nan;
        end
    end
    % %convert azimuth and elevation to rad
    %     curr_az=az(i,:);
    %     curr_az=(curr_az(~isnan(curr_az)));
    %     curr_az=curr_az';
    %     curr_az=deg2rad(curr_az);
    % 
    %     curr_el=el(i,:);
    %     curr_el=(curr_el(~isnan(curr_el)));
    %     curr_el=curr_el';
    %     curr_el=deg2rad(curr_el);
    % 
    %     %3.1 calculate dop
    %     H=[cos(curr_el).*sin(curr_az),cos(curr_el).*cos(curr_az),sin(curr_el),ones(size(curr_az,1),1)];
    %     Q=(H'*H)^-1;
    % 
    %     dops(i,1)=sqrt(Q(1,1)+Q(2,2)+Q(3,3)+Q(4,4)); %GDOP
    %     dops(i,2)=sqrt(Q(1,1)+Q(2,2)+Q(3,3));%PDOP
    %     dops(i,3)=sqrt(Q(1,1)+Q(2,2));%HDOP
    %     dops(i,4)=sqrt(Q(3,3));%VDOP

       % 1 draw sky plot
        % set(sp,AzimuthData=az(1:i,:),ElevationData=el(1:i,:),LabelData=satIDs);
        % drawnow limitrate

end
%% 2 draw satellite visibility
% numSats = numel(satIDs);
% visPlotData = double(vis);
% visPlotData(visPlotData == false) = NaN; % Hide invisible satellites.
% visPlotData = visPlotData + (0:numSats-1); % Add space to satellites to be stacked.
% colors = colororder;
% t=1:epochs_num;
% % set parameters
% figure
% plot(t,visPlotData,".",Color=colors(1,:))
% yticks(1:numSats)
% yticklabels(string(satIDs))
% grid on
% ylabel("GPS Satellite ID")
% xlabel("Time t beginning from 239460(s)")
% title("GPS Satellite Visibility Chart")
% axis tight
%% 4 draw satellite visibility number
numSats = numel(satIDs);
visPlotData = int32(vis);
out=sum(visPlotData,2);
colors = colororder;
t=1:epochs_num;
out=out';
% set parameters

plot(t,out,'.',Color=colors(1,:))
grid on
xlabel("Time t beginning from 239460(s)")
ylabel("number of satellites")
ylim([9, 13]);
set(gca,'YTick',(9:1:13))
title("GPS Satellite Visibility number Chart")

