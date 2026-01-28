function f_est = Poisson_spline(z, k_bin, qq)
%POISSON_SPLINE  Marginal density estimation using Poisson spline regression
%
%   f_est = Poisson_spline(z, k_bin, qq)
%
%   z      : vector of z-values
%   k_bin  : number of bins (if empty, defaults to 100)
%   qq     : quantiles (e.g. [0.25 0.5 0.75]) used for interior knots
%
%   Requires Statistics and Machine Learning Toolbox (glmfit, quantile,
%   normpdf, normcdf, etc.).

    z   = z(:);                 % column vector
    m   = numel(z);
    zMin = min(z);
    zMax = max(z);

    % Number of bins
    if isempty(k_bin)
        K = 100;
    else
        K = k_bin;
    end

    % Bin endpoints and width
    end_point = linspace(zMin, zMax, K + 1);
    d = (zMax - zMin) / K;

    % Bin centers and counts
    y = zeros(K, 1);
    x = zeros(K, 1);

    for i = 1:K
        left  = end_point(i);
        right = end_point(i + 1);
        y(i)  = sum(z >= left & z < right);
        x(i)  = (left + right) / 2;
    end

    % Match R's behavior: add the last point to the last bin
    y(K) = y(K) + 1;

    % Natural cubic spline basis
    kn = quantile(x, qq);              % interior knots
    bb = [min(x) max(x)];              % boundary knots

    B         = natural_cubic_spline_basis(x, kn, bb);
    B_predict = natural_cubic_spline_basis(z, kn, bb);

    % Poisson regression (with intercept)
    % glmfit adds an intercept term automatically
    b = glmfit(B, y, 'poisson', 'constant', 'on');

    % Adjust intercept so that the fitted function is (approximately) a density
    b(1) = b(1) - log(m * d);

    % Density estimate on the original z points
    f_est = exp(b(1) + B_predict * b(2:end));
end


function B = natural_cubic_spline_basis(x, knots, boundary)
%NATURAL_CUBIC_SPLINE_BASIS  Restricted natural cubic spline basis
%
%   B = natural_cubic_spline_basis(x, knots, boundary)
%
%   x         : column vector of points
%   knots     : interior knots (vector)
%   boundary  : [a b], boundary knots
%
%   This implements a standard "restricted cubic spline" basis:
%   column 1  : x
%   columns 2 : K+1  : transformed cubic terms for each interior knot.

    x = x(:);
    a = boundary(1);
    b = boundary(2);
    K = numel(knots);

    % First column: linear term in x
    B = zeros(numel(x), 1 + K);
    B(:,1) = x;

    tp = @(u) max(u, 0).^3;  % truncated power function (.)_+^3

    % Harrell-style restricted cubic spline basis
    for j = 1:K
        k_j = knots(j);

        term1 = tp(x - k_j)    - tp(x - b);
        term2 = tp(x - a)      - tp(x - b);
        coef = (b - k_j) / (b - a);

        B(:, j + 1) = term1 - coef .* term2;
    end
end
