%   XR    = receiver approximate position (X,Y,Z,cdt)4*1
%   XB          = base position (X,Y,Z) 1*3
%   XS           = satellite position (X,Y,Z)  n*3
%   pr_R         = receiver code observations   n
%   pr_B         = base code observations  n

%XR (X,Y,Z,cdt)4*1
%cov_XR  4*4
function [XR, cov_XR, res] = sd_code(XR, XB, XS, pr_R, pr_B)
iter=1;
num=size(XS,1);
res=zeros(1,num);
while iter<=10
    %number of observations
    n = length(pr_R);

    %number of unknown parameters
    m = 4;
    distR2Sat=zeros(n,1);
    for i=1:n
        distR2Sat(i,1)=norm(XS(i,:)-XR(1:3,1)', 'fro');
    end
    
    distB2Sat=zeros(n,1);
    for i=1:n
        distB2Sat(i,1)=norm(XS(i,:)-XB, 'fro');
    end
    
    %design matrix
    %n*4
    A = [((XR(1) - XS(:,1)) ./ distR2Sat), ... %X
         ((XR(2) - XS(:,2)) ./ distR2Sat), ... %Y
         ((XR(3) - XS(:,3)) ./ distR2Sat), ... %z
         ones(n,1)];    %cdt
    
    if any(any(isnan(A))) 
        XR     = NaN(4, 1);
        cov_XR=NaN(4,4);
        return
    end
    
    if any(any(isinf(A))) 
        XR     = NaN(4, 1);
        cov_XR=NaN(4,4);
        return
    end

    cdt=zeros(n,1)+XR(4);
    
    %constant approximate delta pr and cdt
    %n *1
    b = distR2Sat - distB2Sat+cdt;
    
    %observation vector
    %n *1
    y0 = pr_R - pr_B;
    

    if rank(A) ~= 4
        XR     = NaN(4, 1);
        cov_XR=NaN(4,4);
        return
    end

    N=A'*A;
    x  = (N^-1)*A'*(y0-b);
    XR = XR + x;
    
    %estimation of the variance of the observation error
    y_hat = A*x + b;
    v_hat = y0 - y_hat;
    sigma02_hat = (v_hat'*v_hat) / (n-m);
    cov_XR = sigma02_hat * (N^-1);
    res=(y0-b)';
    
   if dot(x,x)<1e-4
        return
    end
end

if iter>Iterations
        XR     = NaN(4, 1);
        cov_XR=NaN(4,4);
end
return