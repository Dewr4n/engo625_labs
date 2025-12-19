% cov: covariance of float ambiguity , n*n
% N_f: float ambiguity， n*1
% N_int : int ambiguity, n*1
function [firstmin, secondmin, firstN, secondN]=Brute_Force_Search(covN, N_f)

n=size(N_f,1);
N=transpose(N_f);
k=6;
firstmin=999999999999;
secondmin=999999999999;
firstN=zeros(1,n);
secondN=zeros(1,n);

nn=zeros(1,n);
nn(1,:)=k;  % k^n 
istr=sym('i',[1,length(nn)]);
str1='';
str2=['[firstmin, secondmin, firstN, secondN]=func(N, covN, firstmin, secondmin, firstN, secondN,',char(istr),')'];
for i=1:length(nn)
    str1=[str1,'for ',char(istr(i)),'=-',num2str(nn(i)/2),':',num2str(nn(i)/2),','];
    str2=[str2,';end'];
end

strFor=[str1,str2]
% tic
eval(strFor)
r=secondmin/firstmin
% timeElapsed = toc
function [firstmin, secondmin, firstN, secondN]=func(N, covN, firstmin, secondmin, firstN, secondN,I)
    N1=N+I;
    N1=round(N1);
    value=(N-N1)*inv(covN)*transpose(N-N1);
    if value<firstmin
        secondmin=firstmin;
        firstmin=value;
        secondN=firstN;
        firstN=N1;
    elseif value< secondmin &&value~=firstmin
        secondmin=value;
        secondN=N1;
    end

end

end