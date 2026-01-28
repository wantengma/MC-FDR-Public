
SaveMaskedData =load("Ross_ap.mat");
ObsData = SaveMaskedData.ObsData;

%%%%%%%%%%% Scaling and projection

r = 100;
scal = 1000;
ObsData(:,3) = (ObsData(:,3)-mean(ObsData(:,3)))/scal;


NewData = SparseLowRProj(ObsData,r);


%% 
%%%%%%%%%%

p = 0.6;
seed = 10000;
rng(seed);
N = size(ObsData,1);
rasmple = randperm(N);

ObsData_1 = ObsData(rasmple(1: round( p*N ) ),:);

ObsData_2 = ObsData(rasmple(round( p*N )+1: round( (1+p)/2 *N ) ),:);

ObsData_3 = ObsData( rasmple(round( (1+p)/2 *N ) +1:end),: );




d1 = length(unique(ObsData(:,1) ) );
d2 = length(unique(ObsData(:,2) ) );


%%%%%%%%%%%%% initialization


r = 30;


%%%%%%%%%%% normalization


M_true = sparse( ObsData(:,1),ObsData(:,2),ObsData(:,3),d1,d2 );

M_obv = sparse( ObsData_1(:,1),ObsData_1(:,2),ObsData_1(:,3),d1,d2 );


[U1,S,V1] = svds(M_obv, r);
M_spec = U1*S*V1';

%%%%%%%%%%% Riemaninit

q = 0.999;

maxiter = 50;
s = 90;



% [M_init,err] =  RiemanInit(M_obv,r,M_spec,maxiter, q,s);
% save('VarM_Ross.mat',"M_init");
Mbackup = load('VarM_Ross.mat');
M_init = Mbackup.M_init;

Delt = abs(M_init(M_true~=0)-M_true(M_true~=0));
sum(Delt<10)/length(Delt)

Display_Riemann = 0;
if Display_Riemann == 1

figure;
plot(1:maxiter,log(err(1:maxiter)) )
xlabel('Iterations')
ylabel('$\log(MSE)$','Interpreter','latex')
title('Convergence of Fast Riemannian Optimization')

tho = 0:0.1:1.5;
proportion_entry = zeros(1,length(tho) );
for i =1:length(tho)
    proportion_entry(i)= sum(Delt<tho(i) )/length(Delt);

end

figure;
plot(tho, proportion_entry)
xlabel('$\delta$','Interpreter','latex')
ylabel('$\# \{ |\Delta M_{ij} | \le \delta \}/N $','Interpreter','latex')
title('Proportion of recovered elements')

end

%%%%%%%%%%%%%%%%% FDR control

%%%%%%%%%% entrywise comparision

%%%%%%%%%% construct Test frame qx3: (i1,j1)  (T or F)

    
q= 20000;
Test= zeros(q,3);
t = 1;
for i =1:d1
    for j = 1:d2

        if (M_true(i,j)~=0)
        Test(t,:) = [i,j, M_true(i,j) ]; %[i,j, M_obv(i,j)*(M_obv(i,j)~=0)+ M_init(i,j)*(M_obv(i,j)==0)  ]; % M_obv(i,j)*(M_obv(i,j)~=0)+ M_init(i,j)*(M_obv(i,j)==0)
        
        flag = t>=q;
        if flag
            break
        end
        t = t+1;
        end
    end
        if flag
            break
        end
end
    


    p=0.1;
    mu = 0.3;
    Rdaveg = 1;
    rng(seed);
    sigHypo = 1; %randi([0,1], Rdaveg*h1*h2, 1)*2 - 1;
     % randomly assign Null vs Non-null

    %Hypo= (1+0.5*mu-mu*rand(q,1)).*sigHypo.*binornd(1,p,q,1);




alpha = 0.1;
%GuessT = zeros(q,1) +1.4; % +0.9

%%%%%%%%%%%%%% Testing with signals choose only one sig to draw the ROC

signalT = 2; %8 nodes->7xdist
    %signalT =2:0.125:2.5;
    epoch = length(signalT);
    
    % FDR = zeros(4,epoch);
    % POW = zeros(4,epoch);


TrueT = Test(:,3);

RankWmulti = zeros(Rdaveg,q);
RankWadd = zeros(Rdaveg,q);
RankWmin = zeros(Rdaveg,q);
RankWBH = zeros(Rdaveg,q);

seed = 10000;
rng(seed);
sigHypo = randi([0,1], q, 1)*2 - 1;
Hypo= (1+0.5*mu-mu*rand(q,1)).*sigHypo.*binornd(1,p,q,1);

for sig = 1:epoch
     
for j = 1:Rdaveg
        
        Test(:,3)= TrueT-signalT(sig)*Hypo;

TData = [ObsData_2;ObsData_3];
N_test = size(TData,1);
TData = TData(randperm(N_test),:);
D1 = TData(1: round(N_test/2),: );
D2 = TData( (round(N_test/2)+1):end,: );

Data = [D1;D2];

    W1 = Feature(M_init,D1,Test,r);
    W2 = Feature(M_init,D2,Test,r);
    W3 = Feature(M_init,Data,Test,r); %GuessT

       truedis =  Hypo~=0; %%%% MT-mT>1
        faldis = Hypo == 0;
        %Data = CreateDataT(M,n);

          
    Drop = 0;
    if Drop==1
    %%%%% drop non-positive
    Z = W1<0 & W2<0;
    W1(Z)=0;
    W2(Z)=0;
    %W3(W3<0)=0;
    %%%%%%%%%
    end   

    Wrank = (W1).*W2;
    % L=Thres(Wrank,alpha)
    % corW = corr(W1(faldis),W2(faldis) )
    Wcomp = sign(W1).*sign(W2).*min(abs(W1),abs(W2));
    Wadd =  sign(W1).*sign(W2).*(abs(W1)+abs(W2));

    RankWmulti(j,:) = Wrank.';
    RankWmin(j,:) = Wcomp.';
    RankWadd(j,:) = Wadd.';

    %%%%% BHq selection by W3
    
    % res1 = Discovery(Wrank,alpha);
    % res2 = Discovery(Wcomp,alpha);
    % res3 = Discovery(Wadd,alpha);

     BH =  2*(1- normcdf( abs(W3)) ) ; %(1- normcdf( (W3)) )    2*(1- normcdf( abs(W3)) )
     RankWBH(j,:) =  -BH.';
    %res4 = BHq(BH,alpha);
    %res5 = Bonferroni(W3,alpha);

    % 
    % 
    %   Td = sum(res1(faldis)==1);
    % Fd = max([sum(res1==1),1])- Td;
    % Dr = Td/max([sum(res1==1),1]);
    % FDR(1,sig) = FDR(1,sig)+ Dr; %[ Td,Fd ,Dr];
    % POW(1,sig) = POW(1,sig)+sum(res1(truedis)==1)/sum(truedis);
    % 
    % Td = sum(res2(faldis)==1);
    % Fd = max([sum(res2==1),1])- Td;
    % Dr = Td/max([sum(res2==1),1]);
    % FDR(2,sig) = FDR(2,sig)+ Dr;    
    % POW(2,sig) = POW(2,sig)+sum(res2(truedis)==1)/sum(truedis);
    % 
    % Td = sum(res3(faldis)==1);
    % Fd = max([sum(res3==1),1])- Td;
    % Dr = Td/max([sum(res3==1),1]);
    % FDR(3,sig) = FDR(3,sig)+ Dr;
    % POW(3,sig) = POW(3,sig)+sum(res3(truedis)==1)/sum(truedis);
    % 
    % 
    % Td = sum(res4(faldis)==1);
    % Fd = max([sum(res4==1),1])- Td;
    % Dr = Td/max([sum(res4==1),1]);
    % FDR(4,sig) = FDR(4,sig)+ Dr; 
    % POW(4,sig) = POW(4,sig)+sum(res4(truedis)==1)/sum(truedis);


%       Td = sum(res5(faldis)==1);
%     Fd = max([sum(res5==1),1])- Td;
%     Dr = Td/max([sum(res5==1),1]);
%     FDR(5,:) = FDR(5,:)+[ Td,Fd ,Dr];
%     POW(5) = POW(5)+sum(res5(truedis)==1)/sum(truedis);


end
end
    % FDR=FDR/Rdaveg
    % POW=POW/Rdaveg



  %% 

  %%%% plot ROC
  [FPmulti, TPmulti]= FindROC(RankWmulti,Hypo);
  [FPmin, TPmin]= FindROC(RankWmin,Hypo);
  [FPadd, TPadd]= FindROC(RankWadd,Hypo);
  [FPBH, TPBH]= FindROC(RankWBH,Hypo);

 figure
 plot(FPmulti, TPmulti,'g-', 'LineWidth', 1)
    hold on
    plot(FPmin, TPmin,'b-', 'LineWidth', 1)
    plot(FPadd, TPadd,'r-', 'LineWidth', 1)
    plot(FPBH, TPBH,'k-', 'LineWidth', 1)
    plot(0:0.01:1, 0:0.01:1,'m--', 'LineWidth', 0.5)
    title('ROC Curve')
    xlabel('False positive rate')
    ylabel('True positive rate (power)')
    legend("multiply", "min", "add","BHq")

  % figure;
  %   subplot(1,2,1)
  %   plot(signalT,FDR(1,:),'g-o',signalT,FDR(2,:),'b-o',signalT,FDR(3,:),'r-o',signalT,FDR(4,:),'k-o')
  %   title('FDR Control')
  %   xlabel('signal')
  %   xlim([min(signalT),max(signalT) ])
  %   ylabel('FDP')
  %   legend("multiply", "min", "add", "BHq")
  %   subplot(1,2,2)
  %   plot(signalT,POW(1,:),'g-o',signalT,POW(2,:),'b-o',signalT,POW(3,:),'r-o',signalT,POW(4,:),'k-o')
  %   title('Power Control')
  %   xlim([min(signalT),max(signalT) ])
  %   xlabel('signal')
  %   ylabel('Power')
  %   legend("multiply", "min", "add", "BHq",'Location','northwest')
  % 
  %   sgtitle('FDP & Power')  %sub-Gaussian, exponential ,student-t

    %     figure;
    % 
    % subplot(1,2,1)
    % plot(signalT,FDR(1,:),'g-o', 'LineWidth', 1)
    % hold on
    % plot(signalT,FDR(2,:),'b-*', 'LineWidth', 1)
    % plot(signalT,FDR(3,:),'r-^', 'LineWidth', 1)
    % plot(signalT,FDR(4,:),'k-+', 'LineWidth', 1)
    % title('FDR Control')
    % xlabel('signal')
    % ylabel('FDP')
    % legend("multiply", "min", "add","BHq")
    % 
    % subplot(1,2,2)
    % plot(signalT,POW(1,:),'g-o', 'LineWidth', 1)
    % hold on
    % plot(signalT,POW(2,:),'b-*', 'LineWidth', 1)
    % plot(signalT,POW(3,:),'r-^', 'LineWidth', 1)
    % plot(signalT,POW(4,:),'k-+', 'LineWidth', 1)
    % title('Power Control')
    % xlabel('signal')
    % ylabel('Power')
    % legend("multiply", "min", "add","BHq",'Location','northwest')
    % 
    % sgtitle('FDP & Power')    %sub-Gaussian, exponential ,student-t

function [FP, TP] =FindROC(W,Hypo)

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

end



function W = Feature(M1,D,T,r)

%%%%% M1 init, D: nx3 ,T:qx3, do the entrywise featuring

[d1,d2]= size(M1);
n = size(D,1);
hatM = M1;


M_obv = sparse( D(:,1),D(:,2),D(:,3),d1,d2 );
Omega = M_obv~=0;

hatM = hatM+ 10*(M_obv-hatM.*Omega);%d1*d2/n*
sigmaX2 = norm(M_obv-M1.*Omega,'fro')^2/n;

[FU,S,FV]=svds(hatM,r);
U=FU(:,1:r);
V=FV(:,1:r);
hatM = U*S*V.';

Uperp = eye(d1)- (U*U.');
Vperp = eye(d2)- (V*V.');


rangeT = size(T,1);


WT = zeros(rangeT,1);


for j =1:rangeT
%         Ej = zeros(d1,d2);
%         Ej(T(j,1), T(j,2))=1;
%         Ej(T(j,3), T(j,4))=-1;
%         Ej = Ej - ( Uperp(:,T(j,1) )* Vperp(T(j,2), :)-Uperp(:,T(j,3) )* Vperp(T(j,4), :) ) ;
% 
% 
%  sT = norm(Ej,'fro');

        s= 1- Uperp(T(j,1),T(j,1))*Vperp(T(j,2),T(j,2));
 zt = hatM(T(j,1),T(j,2))- T(j,3); %- M(T(j,1),T(j,2))+M(T(j,3),T(j,4)
 sign = randi([-1 ,1]);
 WT(j) = ( zt  )/sqrt(s/ n);
end
WT=WT/sqrt(sigmaX2*d1*d2);

W=WT;

end

function w=BHq(q,alpha)
%% w 0 or 1 for rejection

sq = sort(q,"ascend");
m =length(q);
for j =m:-1:1
    if sq(j)<= j/m*alpha
        break
    end
end
L = sq(j);
w = q<=L;


end



function L=Thres(a,alpha)

b = -a(a<0);
na = sum(a>0);
nb = length(a(a<=0));


if isempty(b)
 L=min(a(a>0)); 
 return
% elseif nb/na<= alpha
%     b= sort(b,'descend');
% L = b(1);
% rej = sum(a>=L);
% if rej>= 1/alpha % alpha> 1/rej
%     mu = 0.1;
%     H = 3;
%      alpha = min([alpha,mu*nb/na,H/rej]);
%      for i =b.'
%          L=i;
%          rej = sum(a>=i);
%          rej = max([rej,1]);
%          r = ((sum(a<=-i)+1)/rej);
%     if rej>0  &&  r <= alpha
%      break
%     end
%     end
% else 
%     for i =b.'
%          L=i;
%          rej = sum(a>=i);
%          r = ((sum(a<=-i)+1)/rej);
%     if rej>0  &&  r <= alpha
%      break
%     end
%     end
% end
else 
     mu = 0.016; %0.1, L=i
     ve = 10;
      b=sort(b,"ascend");
 alpha = min([alpha,mu*nb/na  ]);

 for i = b.'
     rej = sum(a>=i) ;
     r = (sum(a<=-i)+1)/rej; %+1
      if rej>0 
      L=i;
      end
     if rej>0 && r<= alpha 
         break
     elseif rej ==1  
         break
 
     end
 
 end

end
    
end

function w=Discovery(W,alpha)

L=Thres(W,alpha);
w = (W>=L);
w= double(w);
    
end 


function z = SparseLowRProj(D,r) %%% D: nx3 sparse matrix, r: rank

M =sparse(D(:,1),D(:,2),D(:,3));
Omega = M~=0;

[U,S,V] = svds(M,r);
Mr = U*S*V';

Mr = Mr.*Omega;

[Z1,Z2,Z3] = find(Mr);

z = [Z1,Z2,Z3];

end
