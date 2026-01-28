function out = efron_local(lfdr_values, alpha)
%EFRON_LOCAL  Efron's local FDR thresholding
%
%   out = efron_local(lfdr_values, alpha)
%
%   lfdr_values : vector of local FDRs
%   alpha       : target FDR level (e.g. 0.1)
%
%   Returns the largest lfdr threshold t such that the estimated
%   FDR of discoveries with lfdr <= t is <= alpha.
%   If no such t exists, returns 0.

    lfdr_values = lfdr_values(:);
    p = numel(lfdr_values);

    % Sort lfdr values
    lfdr_sort = sort(lfdr_values, 'ascend');

    % Cumulative sums and average FDR
    A = cumsum(lfdr_sort);
    B = (1:p)';
    est_fdr = A ./ B;

    % Find largest index with est_fdr <= alpha
    idx = find(est_fdr <= alpha);
    if isempty(idx)
        out = 0;
    else
        k = max(idx);
        out = lfdr_sort(k);
    end
end
