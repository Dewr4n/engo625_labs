%   XR    = receiver approximate position  (X,Y,Z)3*1
%   XB           = base position (X,Y,Z) 1*3
%   XS           = satellite position (X,Y,Z)  n*3
%   ph_R      = rover phase   n*1
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
function [XR, cov_XR, N, cov_N] = dd_phase ...
(XR, XB, XS, ph_R, ph_B, pivot_index,lambda, ...
elevation, pivox_elevation, elevation_rover, pivox_elevation_rover)%,ion_R,ion_B)
iter=1;
    num=length(ph_R);
while iter<=8
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
         diag(-lambda) .* eye(n)];  %ambiguity(zero)
    
    if any(any(isnan(A))) 
        XR     = NaN(3, 1);
        cov_XR=NaN(3,3);
        N=NaN(n,1);
        cov_N=NaN(n,n);
        return
    end
    
    if any(any(isinf(A))) 
        XR     = NaN(3, 1);
        cov_XR=NaN(3,3);
        N=NaN(n,1);
        cov_N=NaN(n,n);
        return
    end
    
    %constant approximate delta pr
    %2n *1
    b = (distR2Sat - distB2Sat)-(distR2Sat(pivot_index)-distB2Sat(pivot_index));
    %b_pr = b + (ion_R  - ion_B)  - (ion_R(pivot_index)   - ion_B(pivot_index));  
    %b_ph = b - (ion_R  - ion_B)  + (ion_R(pivot_index)   - ion_B(pivot_index)); 

    b_ph = b;
    b=b_ph;
    
    %observation vector
    %2n *1
    y0_ph = lambda.*((ph_R - ph_B) - (ph_R(pivot_index) - ph_B(pivot_index)));
    y0=y0_ph;

    %remove pivot-pivot lines
    A( pivot_index, :) = [];
    A(            :, pivot_index+3)       = [];
    b( [pivot_index])    = [];
    y0([pivot_index])    = [];
    n = n - 1;
    
    %weightings
    cov=zeros(n,n);
    for k=1:n
        cov(k,k)=0.01*0.01/sin(elevation(k,1)*pi/180)/sin(elevation(k,1)*pi/180)+0.01*0.01/sin(pivox_elevation*pi/180)/sin(pivox_elevation*pi/180) +...
            0.01*0.01/sin(elevation_rover(k,1)*pi/180)/sin(elevation_rover(k,1)*pi/180)+0.01*0.01/sin(pivox_elevation_rover*pi/180)/sin(pivox_elevation_rover*pi/180);

    end
    P=inv(cov);
    

    NN=A'*P*A;
    x  = (NN^-1)*A'*P*(y0-b);
    
    %estimation of the variance of the observation error
    y_hat = A*x + b;
    v_hat = y0 - y_hat;
    sigma02_hat = (v_hat'*v_hat) / (2*n-m);

    XR = XR + x(1:3,1);

    %estimated double difference ambiguities (without PIVOT)

    N_hat_nopivot = x(4:end);

     %add a zero at PIVOT position
    % N = zeros(n+1,1);
    % N(1:pivot_index-1)   = N_hat_nopivot(1:pivot_index-1);
    % N(pivot_index+1:end) = N_hat_nopivot(pivot_index:end);
    N=N_hat_nopivot;

    Cxx = sigma02_hat * (NN^-1);

    %rover position covariance matrix
    cov_XR = Cxx(1:3,1:3);

    %combined ambiguity covariance matrix
    cov_N_nopivot = Cxx(4:end,4:end);

    %add one line and one column (zeros) at PIVOT position
    % cov_N = zeros(n+1);
    % cov_N(1:pivot_index-1,1:pivot_index-1)     = cov_N_nopivot(1:pivot_index-1,1:pivot_index-1);
    % cov_N(pivot_index+1:end,pivot_index+1:end) = cov_N_nopivot(pivot_index:end,pivot_index:end);
    cov_N = cov_N_nopivot;
    if dot(x(1:3),x(1:3))<1e-4
        return
    end
    iter=iter+1;
end

if iter>10
        XR     = NaN(3, 1);
        cov_XR=NaN(3,3);
        N=NaN(num-1,3);
        cov_N=NaN(num-1,num-1);

end
return