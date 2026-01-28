function [X,Error ]= RiemanInit(X_obs,r,X_init,maxiter, q,s)
% Y:        Observed Signals, n * 1 vector
% A:        Sensing Operators, n*2 
% r:        rank
% X_init:   Initialization, d1 * d2
% X_true:   Ground truth low-rank matrix, need to be recovered, d1 * d2,
%           has rank r
% type:     'noiseless' or 'noisy'. 'noiseless' means there is no noise
% maxiter:  maximum iterations for each phase
% tau:      scale of noise term
% q:        stepsize decay rate, eta_l = eta_0 * q^l
% c:        stepsize in phase two is tau * c / n
% s:        initial stepsize in phase one is s * ||X||_2/n, to approximate
%           \sigma_r


% X:        Estimated low rank matrix
% Error:    a vector of length 2 * maxiter, consists ||X_l - X_true||_F

norm_type = 'fro';

[d1,d2] = size(X_init);
Omega = X_obs~=0;  %%% positions
n = sum(Omega,"all");

%% initialization
X = X_init;
[U,S,V] = svds(X, r);

% l : iteration
l = 0;
%delta0 = norm(X_true , norm_type);
Error = zeros(2*maxiter,1);

hat_sigmar = s* norm(X,2);
        % Initial stepsize
        eta = hat_sigmar/(n);


        
while (l<maxiter)%&&(eta>=0.00001*tau/hat_sigmar)
    l = l + 1;
    %hat_Y = zeros(n,1);
   G = 2* (X.*Omega-X_obs);
    

    
    % G: sub-gradient
    
    % eta: stepsize
    %eta = sum(abs(indc)^2,'all')./norm(G,'fro')^2;
    eta = q * eta;
    %% Riemannian Optimization
    Y1 = (eye(d2) - V * V') * G' * U;
    Y2 = (eye(d1) - U * U') * G * V;
    
    [Q1, R1] = qr(Y1, 0);
    [Q2, R2] = qr(Y2, 0);
    
    U_ = [U, Q2];
    V_ = [V, Q1];
    
    M = [S - eta * U' * G * V,   -eta*R1';  -eta*R2,zeros(r)];
    [U_1, S, V_1] = svds(M, r);
    U = U_*U_1;
    V = V_*V_1;
    X_new = U * S * V';
    
    X = X_new;
    delta =  norm(X.*Omega-X_obs,"fro")^2/n ; %norm(X_new - X_true, norm_type)/delta0;
    Error(l)=delta;
    if mod(l,5)==0
        fprintf('finish round %d\n',l);
    end 

end
Error((l+1):end)=Error(l);        




end