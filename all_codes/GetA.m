%   XR    = receiver approximate position  (X,Y,Z)3*1
%   XB           = base position (X,Y,Z) 1*3
%   XS           = satellite position (X,Y,Z)  n*3
%   pr_R         = rover code observations   n
%   ph_R      = rover phase   n*1
%   pr_B         = base code observations  n
%   ph_B         = base station phase    n
%   pivot_index  = index identifying the pivot satellite
%   elevation    n-1  *1  base station
%   pivox_elevation 1*1 base station
%   elevation_rover  n-1 *1 rover station   
%   pivox_elevation_rover 1*1 rover station


%XR (X,Y,Z)3*1
%cov_XR  3*3
%N combination of ambiguity  (n-1)*1
%cov_N  (n-1)*(n-1)  
function [A, l] = GetA ...
(XR,N, XB, XS, pr_R, ph_R, pr_B, ph_B, pivot_index,lambda)%,ion_R,ion_B)
iter=1;
num=length(pr_R);

while iter<=1
    n = num;
    %number of observations

    
    %number of unknown parameters
    m = 3 + (n - 1);

    distR2Sat=zeros(n,1);
    for i=1:n
        distR2Sat(i,1)=norm(XS(i,:)-XR(1:3,1)', 'fro');
    end
    
    distB2Sat=zeros(n,1);
    for i=1:n
        distB2Sat(i,1)=norm(XS(i,:)-XB, 'fro');
    end
    

    %design matrix(code)
    %A:2n*(3+n)  change to 2n*(3+n-1) later
    A = [((XR(1) - XS(:,1)) ./ distR2Sat) - ((XR(1) - XS(pivot_index,1)) / distR2Sat(pivot_index)), ... %X
         ((XR(2) - XS(:,2)) ./ distR2Sat) - ((XR(2) - XS(pivot_index,2)) / distR2Sat(pivot_index)), ... %Y
         ((XR(3) - XS(:,3)) ./ distR2Sat) - ((XR(3) - XS(pivot_index,3)) / distR2Sat(pivot_index)), ... %z
         zeros(n,n)];  %ambiguity(zero)
    
    A = [A;((XR(1) - XS(:,1)) ./ distR2Sat) - ((XR(1) - XS(pivot_index,1)) / distR2Sat(pivot_index)), ... %X
         ((XR(2) - XS(:,2)) ./ distR2Sat) - ((XR(2) - XS(pivot_index,2)) / distR2Sat(pivot_index)), ... %Y
         ((XR(3) - XS(:,3)) ./ distR2Sat) - ((XR(3) - XS(pivot_index,3)) / distR2Sat(pivot_index)), ... %z
         diag(lambda) .* eye(n)];  %ambiguity(zero)
    
    if any(any(isnan(A))) 

        return
    end
    
    if any(any(isinf(A))) 

        return
    end
    
    b = (distR2Sat - distB2Sat)-(distR2Sat(pivot_index)-distB2Sat(pivot_index));
    b_pr = b;
    b_ph = b;
    b=[b_pr;b_ph];



    y0_pr = (pr_R - pr_B) - (pr_R(pivot_index) - pr_B(pivot_index));
    y0_ph = -lambda.*((ph_R - ph_B) - (ph_R(pivot_index) - ph_B(pivot_index)));
    y0=[y0_pr;y0_ph];
    
    %remove pivot-pivot lines
    A( [pivot_index, pivot_index+n], :) = [];
    A(            :, pivot_index+3)       = [];
    b( [pivot_index, pivot_index+n])    = [];
    y0([pivot_index, pivot_index+n])    = [];

    b(10:end,1)=b(10:end,1)+lambda*N;

    l=y0-b;
    iter=iter+1;
end


return