%   XR    = receiver approximate position  (X,Y,Z)3*1
%   XB           = base position (X,Y,Z) 1*3
%   XS           = satellite position (X,Y,Z)  n*3
%   pr_R         = rover code observations   n
%   ph_R      = rover phase   n*1
%   pr_B         = base code observations  n
%   ph_B         = base station phase    n
%   pivot_index  = index identifying the pivot satellite
%   N            = n-1 ambiguity

%XR (X,Y,Z)3*1
%cov_XR  3*3

function [XR, cov_XR] = dd_code_phase_known_N_cph_only ...
(XR, XB, XS, pr_R, ph_R, pr_B, ph_B, pivot_index,lambda,N)%,ion_R,ion_B)
iter=1;
    num=length(pr_R);
while iter<=10
    n = num;
    %number of observations

    
    %number of unknown parameters
    m = 3;

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
         ((XR(3) - XS(:,3)) ./ distR2Sat) - ((XR(3) - XS(pivot_index,3)) / distR2Sat(pivot_index)) ]; %z
    
    
    if any(any(isnan(A))) 
        XR     = NaN(3, 1);
        cov_XR=NaN(3,3);
        return
    end
    
    if any(any(isinf(A))) 
        XR     = NaN(3, 1);
        cov_XR=NaN(3,3);
        return
    end
    
    %constant approximate delta pr
    %2n *1
    b = (distR2Sat - distB2Sat)-(distR2Sat(pivot_index)-distB2Sat(pivot_index));
    %b_pr = b + (ion_R  - ion_B)  - (ion_R(pivot_index)   - ion_B(pivot_index));  
    %b_ph = b - (ion_R  - ion_B)  + (ion_R(pivot_index)   - ion_B(pivot_index)); 


    %observation vector
    %2n *1
    y0_ph = -lambda.*((ph_R - ph_B) - (ph_R(pivot_index) - ph_B(pivot_index)));
    y0=y0_ph;

    %remove pivot-pivot lines

    A( pivot_index, :) = [];

    b( pivot_index,:)    = [];
    y0(pivot_index,:)    = [];
    n = n - 1;

    b(:,1)=b(:,1)+lambda*N;

        %weightings
    cov=zeros(n,n);
    for k=1:n
        %cov(k,k)=0.3*0.3/sin(elevation(k,1)*pi/180)/sin(elevation(k,1)*pi/180)+0.3*0.3/sin(pivox_elevation*pi/180)/sin(pivox_elevation*pi/180)+ ...
        %0.3*0.3/sin(elevation_rover(k,1)*pi/180)/sin(elevation_rover(k,1)*pi/180)+0.3*0.3/sin(pivox_elevation_rover*pi/180)/sin(pivox_elevation_rover*pi/180);
        
        %cov(n+k,n+k)=0.01*0.01/sin(elevation(k,1)*pi/180)/sin(elevation(k,1)*pi/180)+0.01*0.01/sin(pivox_elevation*pi/180)/sin(pivox_elevation*pi/180) +...
            %0.01*0.01/sin(elevation_rover(k,1)*pi/180)/sin(elevation_rover(k,1)*pi/180)+0.01*0.01/sin(pivox_elevation_rover*pi/180)/sin(pivox_elevation_rover*pi/180);

        cov(k,k)=10.00*10.0;
   
    end
    P=inv(cov);

    NN=A'*P*A;
    x  = (NN^-1)*A'*P*(y0-b);
    residuals=(y0-b);
    %estimation of the variance of the observation error
    y_hat = A*x + b;
    v_hat = y0 - y_hat;
    sigma02_hat = (v_hat'*v_hat) / (2*n-m);

    XR = XR + x(1:3,1);

    %estimated double difference ambiguities (without PIVOT)



     %add a zero at PIVOT position
    % N = zeros(n+1,1);
    % N(1:pivot_index-1)   = N_hat_nopivot(1:pivot_index-1);
    % N(pivot_index+1:end) = N_hat_nopivot(pivot_index:end);

    Cxx = sigma02_hat * (NN^-1);

    %rover position covariance matrix
    cov_XR = Cxx(1:3,1:3);


    %add one line and one column (zeros) at PIVOT position
    % cov_N = zeros(n+1);
    % cov_N(1:pivot_index-1,1:pivot_index-1)     = cov_N_nopivot(1:pivot_index-1,1:pivot_index-1);
    % cov_N(pivot_index+1:end,pivot_index+1:end) = cov_N_nopivot(pivot_index:end,pivot_index:end);

    if dot(x(1:3),x(1:3))<1e-4
        return
    end
    iter=iter+1;
end

if iter>10
        XR     = NaN(3, 1);
        cov_XR=NaN(3,3);

end
return