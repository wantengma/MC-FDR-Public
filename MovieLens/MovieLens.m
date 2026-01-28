%movie lens 100k
FullM = load('ml-100k/u.data');
%
user= max(FullM(:,1));
film=max(FullM(:,2));



%%%%%% Training data

seed = 10000;
rng(seed);



[Ncount, Nuser]=hist(FullM(:,1),  unique(FullM(:,1)) );

[~,Topuser] = sort(Ncount,'descend');

NUser = length(Nuser);

User = Nuser(Topuser(1:NUser));  % ID of users






%%%%%% find the best movies

tempData = FullM(ismember(FullM(:,1),User) ,1:3);

[Ncount, Nmovie]=hist(tempData(:,2),  unique(tempData(:,2)) );

[~,Topmovie] = sort(Ncount,'descend');


NMovie = length(Nmovie);

Movie = Nmovie(Topmovie(1:NMovie));  % ID of Movies



ObsData = tempData(ismember(tempData(:,2),Movie),:);


d1 = NUser;
d2 = NMovie;

N = size(ObsData,1);

for i=1:N
    ObsData(i,1) = find(User==ObsData(i,1));
    ObsData(i,2) = find(Movie==ObsData(i,2));
end
%%%%% unique(ObsData(:,1)) to check

% for i=1:d1
%     ObsData( ObsData(:,1)==User(i),1 )=i;
% end
% 
% for j = 1:d2
%     ObsData( ObsData(:,2)==Movie(j),2 )=j;
% end


%%%%%%%%%%%%% initialization


r = 10;


%%%%%%%%%%% normalization


M_obv = sparse( ObsData(:,1),ObsData(:,2),ObsData(:,3),d1,d2 );



[U1,S,V1] = svds(M_obv, r);
M_spec = U1*S*V1';

%%%%%%%%%%% Riemaninit

q = 0.999;

maxiter = 30;
s = 150;


%[M_init,err] =  RiemanInit(M_obv,r,M_spec,maxiter, q,s);
%save('VarM_Lens.mat',"M_init");
Mbackup = load('VarM_Lens.mat'); % where is this file?
M_init = Mbackup.M_init;

Delt = abs(M_init(M_obv~=0)-M_obv(M_obv~=0));
sum(Delt<0.5)/length(Delt)

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


%%
%%%%%%%%%% shuffle data
rng(seed);
ObsData = ObsData(randperm(N),:);
D1 = ObsData(1: round(N/2),: );
D2 = ObsData( (round(N/2)+1):end,: );

Data = [D1;D2];


%%%%%%%%%% pairwise comparison, 0 is not included: 600 pairs

%%%%%%%%%% construct Test frame qx4: (i1,j1) (i2,j2) (T or F)

Test = [];
q = 1000;
q1 = 300;
t  = 0;
t1 = 0;
for i = 1:d1-1
  for j = 1:d2-1
      if M_obv(i,j)~= 0 && M_obv(i,j+1)~= 0
          if (M_obv(i,j)-M_obv(i,j+1))>+1 && t1<q1
              t = t+1;
              t1 = t1+1;
          Test = [Test; [ i,j,i,j+1,M_obv(i,j)-M_obv(i,j+1) ] ];
          elseif M_obv(i,j)==M_obv(i,j+1)
              t = t+1;
              Test = [Test; [ i,j,i,j+1,M_obv(i,j)-M_obv(i,j+1) ] ];
          end
      end
      flag = size(Test,1)>= q;
      if (flag==1)
         break
      end
   
  end
  if (flag==1)
         break
  end
    
end


[U1,~,V1] = svds(M_init, r);

%%%%%%%%%%%%%% recompute Sigma

% Sig = ProjT(U1,V1,Test);
% save('VarM_Lens.mat',"M_init","Sig");

%%%%%%%%%%%%%% recompute

Mbackup = load('VarM_Lens.mat');
M_init = Mbackup.M_init;
Sigma = Mbackup.Sig;

[Us,S] = schur(Sigma);%+0.0005*eye(q)
Xdesign = Us*  (pinv(sqrt(S),0.001) +0.001*eye(q))*Us.';



Display_W = 1;
if Display_W == 1
GuessT = Test(:,5);
W1 = FeatureSbt(M_init,D1,Test,r,GuessT);
W2 = FeatureSbt(M_init,D2,Test,r,GuessT);

figure;
subplot(1,2,1);
histogram(W1,[-3:0.5:3]);
% Now that a graphics object (histogram plot axes) exists, the following functions will target the
% active axes.
xlabel('Features when $M_T=m_T$','Interpreter','latex')
ylabel('Count')
title('Null Features under D_1')
ax = gca;
ax.FontSize = 18;

subplot(1,2,2);
histogram(W2,[-3:0.5:3]);
% Now that a graphics object (histogram plot axes) exists, the following functions will target the
% active axes.
xlabel('Features when $M_T=m_T$','Interpreter','latex')
ylabel('Count')
title('Null Features under D_2')
ax = gca;
ax.FontSize = 18;

end
%%%%%%%%%%%%%%% perform 1-sided tests and FDR control:
alpha = 0.01;
GuessT = zeros(q,1) +1.4; % +0.9


    FDR = zeros(5,3);
    POW = zeros(5,1);
    W1 = FeatureSbt(M_init,D1,Test,r,GuessT);
    W2 = FeatureSbt(M_init,D2,Test,r,GuessT);
    W3 = FeatureSbt(M_init,Data,Test,r,zeros(q,1)); %GuessT
    

%         Hypok = Hypo( ((k-1)*h1*h2+1) :k*h1*h2);    % Hypok = Hypo( 1 :h1*h2);%
%         Tk = [T1, Hypok];
        truedis =  abs(Test(:,5))> GuessT; %%%% MT-mT>1
        faldis = abs(Test(:,5))<= GuessT;
        %Data = CreateDataT(M,n);

          
    Drop = 1;
    if Drop==1
    %%%%% drop non-positive
    Z = W1<0 & W2<0;
    W1(Z)=0;
    W2(Z)=0;
    %W3(W3<0)=0;
    %%%%%%%%%
    end    
    [Wsda,A]=SDA(D1,D2,Xdesign,Test,r,M_init,GuessT);
    
   
    Wrank = (W1).*W2;
    L=Thres(Wrank,alpha)
    corW = corr(W1(faldis),W2(faldis) )
    Wcomp = sign(W1).*sign(W2).*min(abs(W1),abs(W2));
    Wadd =  sign(W1).*sign(W2).*(abs(W1)+abs(W2));

    %%%%% BHq selection by W3
    
    res1 = Discovery(Wrank,alpha);
    res2 = Discovery(Wcomp,alpha);
    res3 = Discovery(Wadd,alpha);
    res4 = Discovery(Wsda,alpha);

    BH = (1- normcdf( (W3)) ); %(1- normcdf( (W3)) )    2*(1- normcdf( abs(W3)) )
    res5 = BHq(BH,alpha);
    
    Td = sum(res1(faldis)==1);
    Fd = max([sum(res1==1),1])- Td;
    Dr = Td/max([sum(res1==1),1]);
    FDR(1,:) = [ Td,Fd ,Dr];
    POW(1) = sum(res1(truedis)==1)/sum(truedis);
    
    Td = sum(res2(faldis)==1);
    Fd = max([sum(res2==1),1])- Td;
    Dr = Td/max([sum(res2==1),1]);
    FDR(2,:) = [ Td,Fd ,Dr];    
    POW(2) = sum(res2(truedis)==1)/sum(truedis);
    
    Td = sum(res3(faldis)==1);
    Fd = max([sum(res3==1),1])- Td;
    Dr = Td/max([sum(res3==1),1]);
    FDR(3,:) = [ Td,Fd ,Dr];  
    POW(3) = sum(res3(truedis)==1)/sum(truedis);
    
    
    Td = sum(res4(faldis)==1);
    Fd = max([sum(res4==1),1])- Td;
    Dr = Td/max([sum(res4==1),1]);
    FDR(4,:) = [ Td,Fd ,Dr];  
    POW(4) = sum(res4(truedis)==1)/sum(truedis);

    Td = sum(res5(faldis)==1);
    Fd = max([sum(res5==1),1])- Td;
    Dr = Td/max([sum(res5==1),1]);
    FDR(5,:) = [ Td,Fd ,Dr];  
    POW(5) = sum(res5(truedis)==1)/sum(truedis);



%Nuser = randsample(user,300);

% SelectionM = FullM(ismember(FullM(:,1),Nuser) ,1:3);
% User = unique(SelectionM(:,1));
% Film = unique(SelectionM(:,2));
% N  = size(SelectionM,1);
% d1 = length(User);
% d2 = length(Film);
% for i=1:d1
%     SelectionM( SelectionM(:,1)==User(i),1 )=i;
% end
% 
% for j = 1:d2
%     SelectionM( SelectionM(:,2)==Film(j),2 )=j;
% end
% 
% r = 5;
% 
% 
% %N = round(N/3);
% 
% 
% ObsM=SelectionM;
% 
% 
% %%%%%% Construct sensoring
% Y = ObsM(:,3);
% N1=size(ObsM,1);
% X = ObsM(1:N1,1:2);
% M_spec = zeros(d1,d2);
% for i=1:N1
%     E = zeros(d1,d2);
%     E(X(i,1),X(i,2))=1;
%     M_spec =M_spec+ObsM(i,3)*E;
% end
% 
% 
% Y1=Y(1:N1);
% 
% [U1,S,V1] = svds(M_spec, r);
% M_spec = U1*S*V1';
% tau = 1;
% q = 0.9;
% s = 0.1;
% maxiter = 10;
% [M_init,err] =  RiemanInit(Y,X,r,M_spec,maxiter, q,s);
% l=length(err);
% 
% plot(1:l,err)


function W = FeatureSbt(M1,D,T,r,GuessT)

%%%%% M1 init, D: nx3 ,T:qx5, do the substitutation without scaling

[d1,d2]= size(M1);
n = size(D,1);
hatM = M1;


M_obv = sparse( D(:,1),D(:,2),D(:,3),d1,d2 );
Omega = M_obv~=0;

hatM = hatM+ (M_obv-hatM.*Omega);%d1*d2/n*
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

 s = 2-( T(j,1)== T(j,3))*( T(j,2)== T(j,4))-( T(j,3)== T(j,1))*( T(j,4)== T(j,2)) ;
        trcomb = Uperp(T(j,1),T(j,1))* Vperp(T(j,2),T(j,2)) + Uperp(T(j,3),T(j,3))* Vperp(T(j,4),T(j,4))- Uperp(T(j,3),T(j,1))* Vperp(T(j,2),T(j,4))-Uperp(T(j,1),T(j,3))* Vperp(T(j,4),T(j,2)) ; %Uperp(Tk(j,3),Tk(i,1))* Vperp(Tk(i,2),Tk(j,4))-Uperp(Tk(j,1),Tk(i,3))* Vperp(Tk(i,4),Tk(j,2)) ;
        s = s -   trcomb;

 zt = hatM(T(j,1),T(j,2))- hatM(T(j,3),T(j,4)) ; %- M(T(j,1),T(j,2))+M(T(j,3),T(j,4)
 sign = randi([-1 ,1]);
 WT(j) = ( zt -  GuessT(j)*(zt>0)  )/sqrt(s/ n);
end
WT=WT/sqrt(sigmaX2*d1*d2);

W=WT;

end


function S=ProjT(U,V,Tk) %%% Tk px4 M(T1,T2)-M(T3,T4) compute it F-norm after the projection Pm()
d1 = size(U,1);
d2 = size(V,1);
p = size(Tk,1);
S = zeros(p,p);
Uperp = eye(d1)- (U*U.');
Vperp = eye(d2)- (V*V.');

for i=1:p
    for j=i:p 

        s = ( Tk(i,1)== Tk(j,1))*( Tk(i,2)== Tk(j,2))+( Tk(i,3)== Tk(j,3))*( Tk(i,4)== Tk(j,4))-( Tk(i,1)== Tk(j,3))*( Tk(i,2)== Tk(j,4))-( Tk(i,3)== Tk(j,1))*( Tk(i,4)== Tk(j,2)) ;
        trcomb = Uperp(Tk(j,1),Tk(i,1))* Vperp(Tk(i,2),Tk(j,2)) + Uperp(Tk(j,3),Tk(i,3))* Vperp(Tk(i,4),Tk(j,4))- Uperp(Tk(j,3),Tk(i,1))* Vperp(Tk(i,2),Tk(j,4))-Uperp(Tk(j,1),Tk(i,3))* Vperp(Tk(i,4),Tk(j,2)) ; %Uperp(Tk(j,3),Tk(i,1))* Vperp(Tk(i,2),Tk(j,4))-Uperp(Tk(j,1),Tk(i,3))* Vperp(Tk(i,4),Tk(j,2)) ;
        s = s -   trcomb;
        S(i,j)= s;
          S(j,i) = S(i,j);

%         Ei = zeros(d1,d2);
%         Ei(Tk(i,1), Tk(i,2))=1;
%         Ei(Tk(i,3), Tk(i,4))=-1;
%         Ei = Ei - ( Uperp(:,Tk(i,1) )* Vperp(Tk(i,2), :)-Uperp(:,Tk(i,3) )* Vperp(Tk(i,4), :) ) ;
% 
%         Ej = zeros(d1,d2);
%         Ej(Tk(j,1), Tk(j,2))=1;
%         Ej(Tk(j,3), Tk(j,4))=-1;
%         Ej = Ej - ( Uperp(:,Tk(j,1) )* Vperp(Tk(j,2), :)-Uperp(:,Tk(j,3) )* Vperp(Tk(j,4), :) ) ;
%         Ti = bT(i,:); 
%         Tj = bT(j,:);
%         Si = zeros(d1*d2 ,1);
%         Sj = zeros(d1*d2 ,1);
%         for m = 1:d1
%         for n = 1:d2
%             em=zeros(d1,1);
%             em( m)=1;
%             en=zeros(d2,1);
%             en( n )=1;
%             temp = kron(U*U.'*em ,en )+kron(em  ,V*V.'*en )- kron(U*U.'*em,V*V.'*en);
%             Si = Si + Ti((m-1)*d2+n)*temp;
%             Sj = Sj + Tj((m-1)*d2+n)*temp;
%         end
% 
%         end
%          S(i,j)= sum(Ei.*Ej,"all");
%           S(j,i) = S(i,j);

    end
           fprintf('finish the row %d\n',i);
            
end
end
%%%%%%%%%%%%%%% SDA filter
function [w,A]=SDA(D1,D2,X,T1,r,M,GuessT)
% w feature
% A support of LASSO
% Sigma an estimate of T(1-UU)T
% T1-> p*3
% M1-> initialization

% get un-normalized W1,W2:
Dia = DiagST(M,[D1;D2],T1,r);
W1 = unnorm(M,D1,T1,r,GuessT);
W2 = unnorm(M,D2,T1,r,GuessT);
[Y,~] = lasso(X*Dia,X*W1);

%%% find good screening 
truedis = T1(:,5) > GuessT;
p1 = sum(truedis);
for l =1:size(Y,2)
    if sum(Y(truedis,l) ~= 0)/p1 <=0.95
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
drop = 1;
if drop == 1
Z = W1<0 & W2<0;
W1(Z)=0;
W2(Z)=0;
end


Wsda = W1.*W2;
w = Wsda;
A=Ascreen;


end


function W = unnorm(M1,D,T,r,GuessT)

%%%%% M1 init, D: nx3 ,T:qx5, do the substitutation without scaling

[d1,d2]= size(M1);
n = size(D,1);
hatM = M1;

M_obv = sparse( D(:,1),D(:,2),D(:,3),d1,d2 );
Omega = M_obv~=0;

hatM = hatM+ (M_obv-hatM.*Omega); %d1*d2/n*
sigmaX2 = norm(M_obv-M1.*Omega,'fro')^2/n;
[FU,S,FV]=svds(hatM,r);
U=FU(:,1:r);
V=FV(:,1:r);
hatM = U*S*V.';



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
 zt = hatM(T(j,1),T(j,2))- hatM(T(j,3),T(j,4)) ; %- M(T(j,1),T(j,2))+M(T(j,3),T(j,4)
 sign = randi([-1 ,1]);
 WT(j) = ( zt -  GuessT(j)*(zt>0)  )*sqrt(n);
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
     mu = 10; %0.1, L=i
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


function S =DiagST(M1,D,T,r)

[d1,d2]= size(M1);
n = size(D,1);
hatM = M1;

M_obv = sparse( D(:,1),D(:,2),D(:,3),d1,d2 );
Omega = M_obv~=0;

hatM = hatM+ (M_obv-hatM.*Omega); % d1*d2/n*
[FU,~,FV]=svds(hatM,r);
U=FU(:,1:r);
V=FV(:,1:r);
Uperp = eye(d1)- (U*U.');
Vperp = eye(d2)- (V*V.');

rangeT = size(T,1);
dia = zeros(rangeT,rangeT);
for j =1:rangeT
    s = 2-( T(j,1)== T(j,3))*( T(j,2)== T(j,4))-( T(j,3)== T(j,1))*( T(j,4)== T(j,2)) ;
        trcomb = Uperp(T(j,1),T(j,1))* Vperp(T(j,2),T(j,2)) + Uperp(T(j,3),T(j,3))* Vperp(T(j,4),T(j,4))- Uperp(T(j,3),T(j,1))* Vperp(T(j,2),T(j,4))-Uperp(T(j,1),T(j,3))* Vperp(T(j,4),T(j,2)) ; %Uperp(Tk(j,3),Tk(i,1))* Vperp(Tk(i,2),Tk(j,4))-Uperp(Tk(j,1),Tk(i,3))* Vperp(Tk(i,4),Tk(j,2)) ;
        s = s -   trcomb;
 dia(j,j)=sqrt(s);
end
S = dia;

end

function w=Discovery(W,alpha)

L=Thres(W,alpha);
w = (W>=L);
w= double(w);
    
end 

