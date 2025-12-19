function [bcheck, acheck, Qzhat, Qbcheck, bfixed, afixed] = amb_fix(bhat, ahat, Qbb, Qahat, Qba)

%   bhat  = position coordinates (float solution)
%   ahat  = ambiguities (float solution)
%   Qbb   = VCV-matrix (position block)
%   Qahat = VCV-matrix (ambiguity block)
%   Qba   = VCV-matrix (position-ambiguity covariance block)

%   bcheck  = output baseline (fixed or float depending on method and ratio test)
%   acheck  = output ambiguities (fixed or float depending on method and ratio test)
%   Qzhat   = variance-covariance matrix of decorrelated ambiguities
%   Qbcheck = variance-covariance matrix of the baseline
%   bfixed  = output baseline (fixed or float depending on method)
%   afixed  = output ambiguities (fixed or float depending on method)


ratiotest = [];
mutest = [];
succ_rate = [];
fixed_solution = [];


Qbcheck=[];
mu = [];
P0 = 0.001;


try
    bfixed = [];
    % perform ambiguity resolution

        %ILS enumeration (LAMBDA2)
        [U] = chol(Qahat); %compute cholesky decomposition
        Qahat = U'*U; %find back the vcm, now the off diag. comp. are identical
        [afixed,sqnorm,Qzhat,Z,D,L] = lambda_routine2(ahat,Qahat);
        % compute the fixed solution
        bcheck = bhat - Qba*cholinv(Qahat)*(ahat-afixed(:,1));
        Qbcheck = Qbb  - Qba*cholinv(Qahat)*Qba';
        acheck = afixed(:,1);
        % success rate
        Ps = prod(2*normcdf(0.5./sqrt(D))-1);
        %[up_bound, lo_bound] = success_rate(D,L,zeros(length(D)));


catch
    % keep float solution
    bcheck = bhat;
    acheck = ahat;
    Qzhat = Qahat;

    fixed_solution = [fixed_solution 0];
    ratiotest = [ratiotest NaN];
    mutest    = [mutest NaN];
    succ_rate = [succ_rate NaN];

    return
end

if isempty(bfixed)
    bfixed = bcheck;
end

% If IAR_method = 0 or IAR_method = 1 or IAR_method = 2 perform ambiguity validation through ratio test
if (IAR_method == 0)

    if (flag_auto_mu)
        if (1-Ps > P0)
            mu = ratioinv(P0,1-Ps,length(acheck));
        else
            mu = 1;
        end
    end

    ratio = sqnorm(1)/sqnorm(2);

    if ratio > mu
        % rejection; keep float baseline solution
        bcheck = bhat;
        acheck = ahat;

        fixed_solution = [fixed_solution 0];
    else
        fixed_solution = [fixed_solution 1];
    end

    ratiotest = [ratiotest ratio];
    mutest    = [mutest mu];

end

succ_rate = [succ_rate Ps];
