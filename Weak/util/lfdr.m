function out = lfdr(zz, k_bin, qq)
%LFDR  Estimate pi_0 and local FDR
%
%   out = lfdr(zz, k_bin, qq)
%
%   zz     : vector of z-values
%   k_bin  : number of bins (if empty, defaults to 100)
%   qq     : quantiles for Poisson_spline

    zz = zz(:);
    p_total = numel(zz);

    % Two-sided p-values under N(0,1)
    p_value = 2 * normcdf(-abs(zz));
    lambda  = 0.5;

    % Estimate pi_0
    pi_0_est = sum(p_value > lambda) / (p_total * (1 - lambda));

    % Clip pi_0 between lambda and 1 (as in the R intent)
    pi_0_est = max(min(pi_0_est, 1), lambda);

    % Estimate marginal density f(z) via Poisson spline
    f_hat = Poisson_spline(zz, k_bin, qq);

    % Null density f0 (standard normal)
    f0 = normpdf(zz);

    % Local FDR: pi0 * f0 / f_hat
    out = pi_0_est * f0 ./ f_hat;

    % Ensure result is in [0,1]
    out = max(min(out, 1), 0);
end
