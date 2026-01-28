d1 = 400;
d2 = 400;
r = 3;
signalr = 400;
n= 4000;
alpha =0.1;
Rdaveg = 8000;

SampleType = "NoRep+OneSet";

M = CreateLowM(d1,d2,signalr,r);

%%%%save('VarM_noise.mat',"M");
VarM = load('VarM_noise.mat');

M = VarM.M;

W= zeros(Rdaveg,2);
T = [1,1, 5,1];


[FU,S,FV]=svds(M,r);
U=FU(:,1:r);
V=FV(:,1:r);
hatM = U*S*V.';

Uperp = eye(d1)- (U*U.');
Vperp = eye(d2)- (V*V.');

j=1;

Ej = zeros(d1,d2);
Ej(T(j,1), T(j,2))=1;
Ej(T(j,3), T(j,4))=-1;
Ej = Ej - ( Uperp(:,T(j,1) )* Vperp(T(j,2), :)-Uperp(:,T(j,3) )* Vperp(T(j,4), :) ) ;


 sT = norm(Ej,'fro');


seed = 111111;
rng(seed);

for i = 1:Rdaveg
    Data = CreateData(M,n,2,"gaussian");

for k = 1:2
if SampleType == "NoRep+OneSet"
    Dk = Data(k,:,:);
    D = squeeze(Dk);
    n = size(D,1);
    Obv = sparse(D(:,1),D(:,2),D(:,3),d1,d2);
    Omega = Obv ~=0;
    hatM = M + d1*d2/n*Omega.*(Obv-M);
    [FU,S,FV]=svds(hatM,r);
    hatM = FU*S*FV.';

    sigmaX2 = norm(Omega.*(Obv-M),'fro')^2/n;
    j =1;
    zt = hatM(T(j,1),T(j,2))- hatM(T(j,3),T(j,4)) - M(T(j,1),T(j,2))+M(T(j,3),T(j,4));

    W(i,k) = ( zt )/sT;
    W(i,k)=W(i,k)*sqrt(n/sigmaX2/d1/d2);
end

end

end


%% 
subplot(1,2,1);
histogram(normcdf(W(:,1)),'Normalization','probability');
title('p-value of $W_T$','interpreter','latex');
subplot(1,2,2);
histogram(W(:,1).*W(:,2),-3:0.3:3,'Normalization','probability')
title('Symmetricity of $W_T^{(1)}\cdot W_T^{(2)}$','interpreter','latex');



%% 


%%%% draw nature comparison
    q = 200;
    rep = Rdaveg/q;
    alpha = 0.1;

    mu = zeros(q,1);
    q1 = 5;
    rng(seed)
    supp = randsample(q,q1);
    mu(supp) = 1;
    faldis = mu==0;

    signal = 2:0.01:2.9;

    FDR = zeros(2,length(signal));
    
    for sig =1:length(signal)

        for re = 1:rep
            Wmulti = zeros(q,1);
            WBH = zeros(q,1);
            for i = 1:q % construct test stat 

                if mu(i,1) == 1
                    Wmulti(i)= (W( (re-1)*q +i,1)+ signal(sig) )* (W( (re-1)*q +i,2) + signal(sig)) ; 
                    WBH(i) = W( (re-1)*q +i,1)+ signal(sig);

                else

                 Wmulti(i)= (W( (re-1)*q +i,1) )* (W( (re-1)*q +i,2) ) ; 
                 WBH(i) = W( (re-1)*q +i,1);
                end
            end 
            res1 =  Discovery(Wmulti,alpha);
                
            BH = 2*(1- normcdf(abs(WBH)) );
    res2 = BHq(BH,alpha);
                FDR(1,sig) = FDR(1,sig)+sum(res1(faldis)==1)/(max([sum(res1==1),1]) );
    FDR(2,sig) = FDR(1,sig)+sum(res2(faldis)==1)/(max([sum(res2==1),1]) );
        end
    end
    FDR = FDR/rep;
    figure
    plot(signal,FDR(1,:),'g-o', 'LineWidth', 2)
    hold on
    plot(signal,FDR(2,:),'k-+', 'LineWidth', 2)
    xlim([min(signal) max(signal) ])
    title('FDR Control')
    xlabel('signal')
    ylabel('FDP')
        legend("multiply", "BHq")