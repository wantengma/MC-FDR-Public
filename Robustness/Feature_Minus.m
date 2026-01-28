function W = Feature_Minus(M1,D,T,r,M,signalT,SampleType)
%%%% M1 initialization 
%%%% D data n*3
%%%% signal T 1xnsig

[d1,d2]= size(M1);
nsig = length(signalT);

sigmaX2 = 0;

M0 = M1;

if SampleType == "NoRep+OneSet"
    D = squeeze(D);
    n = size(D,1);

        Obv = sparse(D(:,1),D(:,2),D(:,3),d1,d2);
    Omega = Obv ~=0;
    hatM = M1 + d1*d2/n*Omega.*(Obv-M1);
    sigmaX2 = norm(Omega.*(Obv-M1),'fro')^2;

elseif SampleType == "NoRep+TwoSet"

     D1 = squeeze(D(1,:,:));
    D2 = squeeze(D(2,:,:));

    Obv1 = sparse(D1(:,1),D1(:,2),D1(:,3),d1,d2);
    Obv2 = sparse(D2(:,1),D2(:,2),D2(:,3),d1,d2);
    Omega1 = Obv1 ~=0;
    Omega2 = Obv2 ~=0;
        

    Obv2 = Obv2.*(1-Omega1);
    Obv = Obv1+Obv2;
    Omega = Obv ~=0;
    n = sum(Omega,"all");


      hatM = M0 + d1*d2/n*Omega1.*(Obv1-M1) + d1*d2/n*Omega2.*(Obv2-M1);
    sigmaX2 = norm(Omega1.*(Obv1-M1),'fro')^2+ norm(Omega2.*(Obv2-M1),'fro')^2;
else
    % for i =1:n
    %     e1=zeros(d1,1);
    %     e1(D(i,1))=1;
    %     e2=zeros(d2,1);
    %     e2(D(i,2))=1;
    %     hatM = hatM+ d1*d2/n* e1* e2.'*( D(i,3)- M1(D(i,1),D(i,2)) );
    %      sigmaX2 = sigmaX2+1*( D(i,3)- M1(D(i,1),D(i,2)) )^2;
    % end
end

sigmaX2=sigmaX2/n;
[FU,S,FV]=svds(hatM,r);
U=FU(:,1:r);
V=FV(:,1:r);
hatM = U*S*V.';

Uperp = eye(d1)- (U*U.');
Vperp = eye(d2)- (V*V.');


rangeT = size(T,1);


WT = zeros(rangeT,nsig);


for j =1:rangeT
        Ej = zeros(d1,d2);
        Ej(T(j,1), T(j,2))=1;
        Ej(T(j,3), T(j,4))=-1;
        Ej = Ej - ( Uperp(:,T(j,1) )* Vperp(T(j,2), :)-Uperp(:,T(j,3) )* Vperp(T(j,4), :) ) ;


 sT = norm(Ej,'fro');
 zt = hatM(T(j,1),T(j,2))- hatM(T(j,3),T(j,4)) - M(T(j,1),T(j,2))+M(T(j,3),T(j,4));
 
 WT(j,:) = ( zt +  signalT*T(j,5)  )/sT;
end
WT=WT*sqrt(n/sigmaX2/d1/d2);

W=WT;

end