function M = CreateLowM(d1,d2,sig,r,seed)

if nargin < 4
seed = 111111;
end
rng(seed);

A=randn(d1,r);
B=randn(r,d2);
[U,~,~]=svds(A,r);
X = U;
[~,~,V]=svds(B,r);
Y =V;
M= X*diag(repelem(sig,r) )*Y.';

end