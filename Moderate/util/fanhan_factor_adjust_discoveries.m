function [disc, t_star, fdp_hat_t, t_grid, fdp_grid] = fanhan_factor_adjust_discoveries(W, SigmaHat, alpha, k)
% FANHAN_FACTOR_ADJUST_DISCOVERIES
%   Factor-adjusted FDP control following Fan & Han (2017).
%
% INPUTS:
%   W         : n x 1 vector of test statistics (asymptotically N(0,1) under H0)
%   SigmaHat  : n x n estimated covariance matrix of W
%   alpha     : target FDP/FDR level (e.g., 0.1)
%   k         : (optional) number of factors; if omitted, chosen by explained variance
%
% OUTPUTS:
%   disc      : n x 1 vector, 1 = reject (discovery), 0 = no reject
%   t_star    : selected two-sided p-value threshold
%   fdp_hat_t : FDP estimate at t_star
%   t_grid    : grid of thresholds used
%   fdp_grid  : estimated FDP for each t in t_grid
%
% Requires Statistics and Machine Learning Toolbox (normcdf, norminv).

    if nargin < 3 || isempty(alpha)
        alpha = 0.1;
    end

    W = W(:);
    n = length(W);
    

    % 1) Build correlation matrix from SigmaHat
    % Before everything else:
    SigmaHat = (SigmaHat + SigmaHat.') / 2;
    d = sqrt(diag(SigmaHat));
    W = W(:) ./ d;             % standardize to variance 1
    D = d * d.';
    Rhat = SigmaHat ./ D;      % correlation matrix


    % 2) Eigen-decomposition of correlation matrix
    [V, Dlam] = eig((Rhat + Rhat.') / 2);
    lambda = diag(Dlam);
    [lambda, idx] = sort(lambda, 'descend');
    V = V(:, idx);

    % 3) Choose number of factors k (if not provided)
    if nargin < 4 || isempty(k)
        cumvar = cumsum(lambda) / sum(lambda);
        k = find(cumvar >= 0.6, 1, 'first');   
        if isempty(k)
            k = min(10, n);
        end
    end

    k = min(k, n);
    lambdak = lambda(1:k);
    Vk = V(:, 1:k);

    % 4) Factor loadings matrix B (n x k)
    B = Vk * diag(sqrt(lambdak));  % each column j: sqrt(lambda_j) * v_j

    % 5) Factor scores (k x 1)
    BtB = B.' * B;
    Fhat = BtB \ (B.' * W);        % least squares: (B'B)^(-1) B' W

    % 6) Residual variance and eta_i
    Bi2    = sum(B.^2, 2);            % ||b_i||^2 for each i
    sigma2 = 1 - Bi2;
    sigma2 = max(sigma2, 1e-6);       % numerical safeguard
    sigma  = sqrt(sigma2);            % this corresponds to inv_a in the R code
    
    eta    = B * Fhat;                % factor part (analog of bW.est)


    % 7) Grid of p-value thresholds t
    t_min = 1e-4;
    t_max = 0.5;                  % typical upper bound for BH-like procedures
    m_grid = 200;
    t_grid = linspace(t_min, t_max, m_grid);

    Ahat = zeros(size(t_grid));    % estimated # of false discoveries (upper bound)
    R = zeros(size(t_grid));       % number of rejections

    absW = abs(W);

    for j = 1:m_grid
        t = t_grid(j);
    
        % two-sided p-value t -> quantile of one tail
        qtl = norminv(t/2);           % this is negative; in R they use this directly
    
        % Approximate # of false discoveries
        term1 = normcdf((qtl + eta) ./ sigma);
        term2 = normcdf((qtl - eta) ./ sigma);
        Ahat(j) = sum(term1 + term2);
    
        % Rejections at threshold t (two-sided on W)
        z_thr = norminv(1 - t/2);
        R(j) = sum(absW >= z_thr);
    end


    % 8) FDP estimate and choice of threshold
    R_pos = max(R, 1);             % avoid division by zero
    fdp_grid = min(1, Ahat ./ R_pos);

    % valid thresholds: estimated FDP <= alpha and at least one rejection
    valid = (fdp_grid <= alpha) & (R > 0);

    if any(valid)
        % choose the largest such t (most discoveries for given alpha)
        j_star = find(valid, 1, 'last');
        t_star = t_grid(j_star);
        fdp_hat_t = fdp_grid(j_star);
    else
        % no threshold meets the FDP target; return no discoveries
        t_star = t_min;
        fdp_hat_t = 1;
        disc = zeros(n, 1);
        return;
    end

    % 9) Final discovery set at t_star
    z_star = norminv(1 - t_star/2);
    disc = double(absW >= z_star);
end
