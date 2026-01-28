d1 = 1000;
d2 = 1000;
r = 3;
signalr = 1000;
n= 4800;
alpha = 0.1;


Rdaveg = 40;

M = CreateLowM(d1,d2,signalr,r);
%save('VarM.mat',"M");

VarM = load('VarM.mat');
M = VarM.M;

    
    
seed = 111111;
rng(seed);
Data= CreateData(M,n,2*Rdaveg,"gaussian");

%%%% resample 1
% save('VarD.mat',"Data")
% VD = load('VarD.mat');
% Data = VD.Data;
% 
% SDA_data = load('SDA-data.mat');
% 
% Data = SDA_data.Data;

      %%%% 
    %%%% 
    %%%% 
    %%%% SDA for a row:
    h1 = 1;
    h2 = 400;
    T1 = zeros(h1*h2,4); % tests
    for i= 1:h1
        for j = 1:h2
            T1( (i-1)*h2+j,: ) = [i,j,2,j];
        end 
    end
    
    % randomly assign Null vs Non-null multiple round 
    p=0.2;
    mu = 0.3;
    %Rdaveg = 10;
    sigHypo = randi([0,1], Rdaveg*h1*h2, 1)*2 - 1;
    Hypo= (1+0.5*mu-mu*rand(Rdaveg*h1*h2,1)).*sigHypo.*binornd(1,p,Rdaveg*h1*h2,1);
    
    
    save('VarT2.mat',"Hypo");
    VarT = load('VarT2.mat');
    Hypo = VarT.Hypo;
    [U,~,V]=svds(M,r);

%      Sigma = ProjT(U,V,T1);
%     save('S_rowdiff.mat',"Sigma");
    Sig = load('S_rowdiff.mat');
%     Sig = load('S.mat');
    Sigma = Sig.Sigma;
    X = inv(sqrtm(Sigma) );
    Scov = inv(diag(sqrt(diag(Sigma)) ))*Sigma*inv(diag( sqrt(diag(Sigma)) ));

    
     signalT = 2;
    
    
    epoch = length(signalT);
    
    
    

    
    FDR = zeros(5,epoch);
    POW = zeros(5,epoch);

    q = h1*h2;
RankWmulti = zeros(Rdaveg,q);
RankWadd = zeros(Rdaveg,q);
RankWmin = zeros(Rdaveg,q);
RankWSDA = zeros(Rdaveg,q);
RankWBH = zeros(Rdaveg,q);
    
    for k=1:Rdaveg
       
        
        Hypok = Hypo( 1 :h1*h2);% Hypok = Hypo( ((k-1)*h1*h2+1) :k*h1*h2); 
        Tk = [T1, Hypok];
       
        truedis = Hypok~=0;
        faldis = Hypok==0;
        D1 =Data(k,:,:);
        D2 =Data( k+Rdaveg,:,:);
        Dall = [D1;D2];

    W1all = Feature_Minus(M,D1,Tk,r,M,signalT,"NoRep+OneSet"); %W1: T x sig
    W2all = Feature_Minus(M,D2,Tk,r,M,signalT,"NoRep+OneSet");
    W3all = Feature_Minus(M,Dall,Tk,r,M,signalT,"NoRep+TwoSet"); 

    for sig = 1:epoch
    W1 = W1all(:,sig);
    W2 = W2all(:,sig);
    W3 = W3all(:,sig);
    [Wsda,A]=SDA(W1,W2,X,Sigma,Tk);
     
    Drop = 0;
    if Drop==1
    %%%%% drop non-positive
    Z = W1<0 & W2<0;
    W1(Z)=0;
    W2(Z)=0;
    %%%%%%%%%
    end    

    
   
    Wrank = (W1).*W2;
    L=Thres(Wrank,alpha)
    corW = corr(W1(faldis),W2(faldis) )
    Wcomp = sign(W1).*sign(W2).*min(abs(W1),abs(W2));
    Wadd =  sign(W1).*sign(W2).*(abs(W1)+abs(W2));
    BH = 2*(1- normcdf(abs(W3)) );

    RankWmulti(k,:) = Wrank.';
RankWadd(k,:)  = Wadd.';
RankWmin(k,:)  = Wcomp.';
RankWSDA(k,:)  = Wsda.';
RankWBH(k,:)  = -BH.';

    end

    end

%% 
ROCmulti = zeros(2,q);
ROCmin = zeros(2,q);
ROCadd = zeros(2,q);
ROCBH = zeros(2,q);
ROCSDA = zeros(2,q);

for k = 1:Rdaveg
    ROCmulti = ROCmulti+FindROC(RankWmulti(k,:),Hypok);
    ROCmin = ROCmin+FindROC(RankWmin(k,:),Hypok);
    ROCadd = ROCadd+FindROC(RankWadd(k,:),Hypok);
    ROCBH = ROCBH+FindROC(RankWBH(k,:),Hypok);
    ROCSDA = ROCSDA+FindROC(RankWSDA(k,:),Hypok);

end
ROCmulti = ROCmulti/Rdaveg;
ROCmin=ROCmin/Rdaveg;
ROCadd=ROCadd/Rdaveg;
ROCBH=ROCBH/Rdaveg;
ROCSDA=ROCSDA/Rdaveg;

  %%%% plot ROC
  % [FPmulti, TPmulti]= FindROC(RankWmulti,Hypo);
  % [FPmin, TPmin]= FindROC(RankWmin,Hypo);
  % [FPadd, TPadd]= FindROC(RankWadd,Hypo);
  % [FPBH, TPBH]= FindROC(RankWBH,Hypo);

 figure
 plot(ROCmulti(1,:), ROCmulti(2,:),'g-', 'LineWidth', 1)
    hold on
    plot(ROCmin(1,:), ROCmin(2,:),'b-', 'LineWidth', 1)
    plot(ROCadd(1,:), ROCadd(2,:),'r-', 'LineWidth', 1)
    plot(ROCSDA(1,:),ROCSDA(2,:) ,'c-','LineWidth',1)
    plot(ROCBH(1,:), ROCBH(2,:),'k-', 'LineWidth', 1)
    plot(0:0.01:1, 0:0.01:1,'m--', 'LineWidth', 0.5)
    title('ROC Curve')
    xlabel('False positive rate')
    ylabel('True positive rate (power)')
        legend("multiply", "min", "add", "SDA","BHq")
    
    
    
function z =FindROC(W,Hypo)

[~, perm]= sort(W,"ascend");
PHypo = Hypo ~=0;
PHypo = PHypo(perm); 
%AltHypo = Hypo ~=0;
N = length(W);
p1 = sum(PHypo);
p0 = N-p1;

FPrem = p0;
TPrem = p1;

FP = zeros(1,N); 
TP = zeros(1,N);
% first row:   false positive FP/p0
% second row: true positive TP/p1
for i = 1:N
    FP(i) = FPrem/p0;
    TP(i)  = TPrem/p1;
    if PHypo(i)
       TPrem = TPrem -1;
    else
        FPrem = FPrem -1;
    end    

end
z= [FP;TP];

end