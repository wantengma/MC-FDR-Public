function W = Feature(M1,D,T,r,signalT,Tshape,DataGroup)
%%%% M1: ground true or hypothesized value
%%%% signalT: the hypothesized values
%%%% D: data n*3 (if it is double, then n = 2*n)
%%%% T: linear form (2 or 5 columns, the last column is H_1=1/0)
%%%% signalT: 1xnsig
%%%% Tshape: "entry", "two_diff"
%%%% Datagroup: 2n*3, used for sample splitting


if nargin < 7 % if the last input is not provided
    DataGroup = "single"; % Default value for b
end

[d1,d2]= size(M1);


if DataGroup == "single"
    D = squeeze(D);
    Obv = sparse(D(:,1),D(:,2),D(:,3),d1,d2);

elseif DataGroup == "double"

    D1 = squeeze(D(1,:,:));
    D2 = squeeze(D(2,:,:));
    Obv1 = sparse(D1(:,1),D1(:,2),D1(:,3),d1,d2);
    Obv2 = sparse(D2(:,1),D2(:,2),D2(:,3),d1,d2);
    Omega1 = Obv1 ~=0;
    Obv2 = Obv2.*(1-Omega1);
    Obv = Obv1+Obv2;
end

% In both cases, we have observed data and Omega
Omega = Obv ~=0;
n = sum(Omega,"all");


%%% Gradient Descent (GD)
GDinit = 1;
[U0,S0,V0] = svds(M1,r);
M0 = U0*S0*V0';


if GDinit

% Initial estimate for GD
Mobv = d1*d2/n*Omega.*(Obv); % 
[U0,S0,V0] = svds(Mobv,r);
X = U0* sqrt(S0);
Y = V0* sqrt(S0);
M0 = X*Y';
% parameters used for gradient descent
q = 0.999;
maxiter = 10;
s = 60;
M0 = RiemanInit(Obv,r,M0,n,maxiter,q,s);
end


% debiased approach
%M0 = M1; % use the true one
hatM = M0 + d1*d2/n*Omega.*(Obv-M0);
sigmaX2 = norm(Omega.*(Obv-M0),'fro')^2;
sigmaX2 = sigmaX2/n;
% SVD for initial estimate
[U_int,~,V_int] = svds(M0,r);


% the new projection method
project_m = 0;
if project_m == 1

    [FU,~,~]=svds(hatM*V_int,r);
    U=FU(:,1:r);
    [~,~,FV]=svds(U_int'*hatM,r);
    V=FV(:,1:r);
    hatM = U*(U')*hatM*V*(V');

else % the old projection method

    [U_hat, ~, V_hat] = svds(hatM,r);
    hatM = U_hat*(U_hat')*hatM*V_hat*(V_hat');

end



% Creat the test statistics
Uperp = eye(d1)- (U_int*U_int.');
Vperp = eye(d2)- (V_int*V_int.');

rangeT = size(T,1);
nsig = length(signalT);
WT = zeros(rangeT,nsig);

if Tshape == "entry"

    for j =1:rangeT
        Ej = zeros(d1,d2);
        Ej(T(j,1), T(j,2))=1;
        Ej = Ej - ( Uperp(:,T(j,1) )* Vperp(T(j,2), :) ) ;

        sT = norm(Ej,'fro');
        zt = hatM(T(j,1),T(j,2)) - M1(T(j,1),T(j,2));
        WT(j,:) = ( zt +  signalT*T(j,3)  )/sT;
    end

elseif Tshape == "two-diff"

    for j =1:rangeT
        Ej = zeros(d1,d2);
        Ej(T(j,1), T(j,2))=1;
        Ej(T(j,3), T(j,4))=-1;
        Ej = Ej - ( Uperp(:,T(j,1) )* Vperp(T(j,2), :)-Uperp(:,T(j,3) )* Vperp(T(j,4), :) ) ;


        sT = norm(Ej,'fro');
        zt = hatM(T(j,1),T(j,2))- hatM(T(j,3),T(j,4)) - M1(T(j,1),T(j,2))+M1(T(j,3),T(j,4));
        WT(j,:) = ( zt +  signalT*T(j,5)  )/sT;
    end
end

WT=WT*sqrt(n/sigmaX2/d1/d2);

W = full(WT);

end