%yandi 10/7/2023
%sat_pos in ecef x;y;z....
%psudoranges in meter 1 2 3...
%pos ecef x;y;z;dt
%res PR residuals
function [pos,res,cov] = SPP(sat_pos,ranges)
pos= zeros(4, 1);%x;y;z;dt
cov=NaN(4,4);
Iterations = 10;
num=size(sat_pos,2);%satellite number
res=zeros(1,num);
%calculate design matrix
A=zeros(num, 4);
omc= zeros(num, 1);
iter=1;
while iter<=Iterations
    %calculate design matrix and residual
    for i=1:num
    	A(i, :) =  [ (-(sat_pos(1,i) - pos(1))) / ranges(i) ...
        (-(sat_pos(2,i) - pos(2))) / ranges(i) ...
        (-(sat_pos(3,i) - pos(3))) / ranges(i) ...
        1 ];
        %residual 
        omc(i) = (ranges(i) - norm(sat_pos(:,i) - pos(1:3), 'fro') - pos(4));
    end
    if any(any(isnan(A))) 
        pos     = NaN(4, 1);
        cov=NaN(4,4);
        return
    end
    
    if any(any(isinf(A))) 
        pos     = NaN(4, 1);
        cov=NaN(4,4);
        return
    end
    
    for ii=1:4
        for jj=1:4
            if isnan(A(ii,jj))||isinf(A(ii,jj))
                return
            end

        end
    end
    
    if rank(A) ~= 4
        pos     = NaN(4, 1);
        cov=NaN(4,4);
        return
    end
    
    %update
    x   = (A'*A)^-1*A'*omc;
    pos = pos + x;
    cov=(A'*A)^-1;
    
    %get residuals
    res=omc';
    if dot(x,x)<1e-4
        return
    end
end
if iter>Iterations
        pos     = NaN(4, 1);
        cov=NaN(4,4);
end
return