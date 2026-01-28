function [reject, lfdr_vals, lfdr_thr] = local_fdr_procedure(z, alpha, k_bin, qq)
%LOCAL_FDR_PROCEDURE  One-stop local FDR testing using Efron's rule.
%
%   [reject, lfdr_vals, lfdr_thr] = local_fdr_procedure(z, alpha, k_bin, qq)
%
%   Inputs
%   ------
%   z      : vector of z-scores (test statistics, assumed N(0,1) under H0)
%   alpha  : desired FDR level (e.g. 0.1). If empty or omitted, defaults to 0.1.
%   k_bin  : number of bins for Poisson spline density estimate. If empty or
%            omitted, defaults to 100.
%   qq     : quantiles for spline interior knots, e.g. [0.25 0.5 0.75].
%            If empty or omitted, defaults to [0.25 0.5 0.75].
%
%   Outputs
%   -------
%   reject    : logical vector, reject(i) = true if hypothesis i is rejected
%               (i.e. lfdr_i <= lfdr_thr)
%   lfdr_vals : estimated local FDR values for each z(i)
%   lfdr_thr  : lfdr threshold chosen by Efron's local FDR rule
%
%   Example
%   -------
%       % z-scores:
%       z = randn(1000,1);
%       % run local FDR at level 0.1
%       [reject, lfdr_vals, lfdr_thr] = local_fdr_procedure(z, 0.1);
%

    % Ensure column vector
    z = z(:);

    % Defaults
    if nargin < 2 || isempty(alpha)
        alpha = 0.1;
    end
    if nargin < 3 || isempty(k_bin)
        k_bin = 100;
    end
    if nargin < 4 || isempty(qq)
        qq = [0.25 0.5 0.75];
    end

    % 1) Compute local FDR values
    lfdr_vals = lfdr(z, k_bin, qq);

    % 2) Get Efron's local FDR threshold
    lfdr_thr = efron_local(lfdr_vals, alpha);

    % 3) Rejection rule: lfdr_i <= threshold
    reject = lfdr_vals <= lfdr_thr;
end
