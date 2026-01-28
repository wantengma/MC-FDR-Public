function w = Discovery(a, alpha)
% DISCOVERY  Thresholding-based discovery rule
%   w = Discovery(a, alpha)
%   a     : test statistics (vector)
%   alpha : target level
%   w     : 0/1 discovery vector


    b  = -a(a < 0);
    na = sum(a > 0);
    nb = length(a(a <= 0));

    % ---------- Compute threshold L (inlined Thres) ----------
    if isempty(b)
        % No negative values: threshold is min positive
        if any(a > 0)
            L = min(a(a > 0));
        else
            % No positive or negative values -> no discoveries
            L = Inf;
        end
    else
        mu = 0.1;         % unused but kept for consistency with original
        b  = sort(b, "ascend");

        L = Inf;          % default, will be updated in loop if rej>0
        n = length(b);
        for  z = 1:n
            i = b(z);
            rej = sum(a >= i);
            if rej > 0
                L =  b(ceil(max(z-1,1)) );
            else
                continue;
            end

            r = sum(a <= -L) / (rej+1);   % (+1) removed 

            if rej > 0 && r <= alpha
                break
            elseif rej == 1
                break
            end
        end
    end

    % ---------- Apply threshold ----------
    w = double(a >= L);
end
