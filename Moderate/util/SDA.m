
function [w,A]=SDA(W1,W2,X,Sigma,T1)
% w feature
% A support of LASSO
% Sigma an estimate of T(1-UU)T
% T1-> px5
% X -> inverse = \Simga^{-1/2}
% M-> initialization

% get un-normalized W1,W2:
Dia = diag(sqrt(diag(Sigma)));

W1 = Dia*W1;
W2 = Dia*W2;
[Y,~] = lasso(X*Dia,X*W1);

%%% find good screening 
truedis = T1(:,5) ~=0;
p1 = sum(truedis);
for l =1:size(Y,2)
    if sum(Y(:,l) ~= 0)/p1 <=1.2
        break
    end
end
 l = max(2,l-5);
Ascreen =  Y(:, l)~= 0;

Xsc = X(:,Ascreen);
Wsc = (Xsc.'*Xsc)\Xsc.'* X*W2;
Wreg = zeros(size(X,1),1);
Wreg(Ascreen)= sqrt(diag(diag( inv(Xsc.'*Xsc) ) ))\ Wsc;

W1 = Y(:,l);
W2 = Wreg;

Drop = 1;
if Drop == 1
    Z      = (W1 < 0) & (W2 < 0);
    W1(Z)  = 0;
    W2(Z)  = 0;
end

Wsda = W1.*W2;



w = Wsda;
A=Ascreen;

end
