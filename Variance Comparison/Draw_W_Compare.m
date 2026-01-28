d1 = 800;
d2 = 800;
r = 5;
signalr = 800;
n= ceil(3600 * 5/3*8/4); %ceil(3600 * 5/3)
alpha =0.1;
Rdaveg = 5000;

SampleType = "NoRep+OneSet";

seed = 1111111;
M = CreateLowM(d1,d2,signalr,r,seed);

% %%%%save('VarM_noise.mat',"M");
% VarM = load('VarM_noise.mat');
% 
% M = VarM.M;

W= zeros(Rdaveg,2);
% Wcomp= zeros(Rdaveg,1);
%T = [1,1, 5,1];

T = zeros(d1,d2);
T(1,1)=1;
T(2,1)=1;
T(5,1)=-1;
T(1,10)=-1;

MT = sum(M.*T,"all");

[FU,S,FV]=svds(M,r);
U=FU(:,1:r);
V=FV(:,1:r);
hatM = U*S*V.';

Uperp = eye(d1)- (U*U.');
Vperp = eye(d2)- (V*V.');

j=1;

% Ej = zeros(d1,d2);
% Ej(T(j,1), T(j,2))=2;
% Ej(T(j,3), T(j,4))=-1;
Ej = T -Uperp*T*Vperp;
%Ej = Ej - ( Uperp(:,T(j,1) )* Vperp(T(j,2), :)-Uperp(:,T(j,3) )* Vperp(T(j,4), :) ) ;


 sT = norm(Ej,'fro');

 sTcomp = sqrt(norm(U.'*T,'fro')^2 + norm(T*V,'fro')^2);


seed = 111111;
rng(seed);



parfor i = 1:Rdaveg

    Data = CreateData(M,n,2,"gaussian",seed*i);

for k = 1:2
if SampleType == "NoRep+OneSet"
    Dk = Data(k,:,:);
    D = squeeze(Dk);
    nsize = size(D,1);
    Obv = sparse(D(:,1),D(:,2),D(:,3),d1,d2);
    Omega = Obv ~=0;
    hatM = M + d1*d2/nsize*Omega.*(Obv-M);
    sigmaX2 = norm(Omega.*(Obv-M),'fro')^2/n;
    [FU,S,FV]=svds(hatM,r);
    hatM = FU*S*FV.';

    
    j =1;
    zt =  sum(hatM.*T,"all")- MT; %hatM(T(j,1),T(j,2))- hatM(T(j,3),T(j,4)) - M(T(j,1),T(j,2))+M(T(j,3),T(j,4));

    W(i,k) = ( zt );
    W(i,k)=W(i,k)*sqrt(n/sigmaX2/d1/d2);
    % 
    % Wcomp(i,k) = ( zt )/sTcomp;
    % Wcomp(i,k)=Wcomp(i,k)*sqrt(n/sigmaX2/d1/d2);


end

end

end
W = W./[sT, sTcomp];
% 
% for i = 1:Rdaveg
%     Data = CreateData(M,n,2,"gaussian");
% 
% for k = 1:1
% if SampleType == "NoRep+OneSet"
%     Dk = Data(k,:,:);
%     D = squeeze(Dk);
%     n = size(D,1);
%     Obv = sparse(D(:,1),D(:,2),D(:,3),d1,d2);
%     Omega = Obv ~=0;
%     hatM = M + d1*d2/n*Omega.*(Obv-M);
%     [FU,S,FV]=svds(hatM,r);
%     hatM = FU*S*FV.';
% 
%     sigmaX2 = norm(Omega.*(Obv-M),'fro')^2/n;
%     j =1;
%     zt =  sum(hatM.*T,"all")- MT; %hatM(T(j,1),T(j,2))- hatM(T(j,3),T(j,4)) - M(T(j,1),T(j,2))+M(T(j,3),T(j,4));
% 
%     Wcomp(i,k) = ( zt )/sTcomp;
%     Wcomp(i,k)=Wcomp(i,k)*sqrt(n/sigmaX2/d1/d2);
% 
% end
% 
% end
% 
% end

% save simulation data as


save("Data_W/d="+string(d1)+"-r="+string(r)+"-n="+string(n)+".mat","W");

%% 
D = load("good d=800-r=5-n=12000.mat");
W = D.W;
%%%%%%%%%%%%%%% read the simulation results and draw plots

z = -0.6:0.02:0.6; % plot the empirical distribution function
lz = length(z);

Wcomp = W(:,2);
Wour = W(:,1);

FW = zeros(1,lz); 
FWcomp = zeros(1,lz); 

for i = 1:lz
FW(i) = mean(Wour<z(i));
FWcomp(i) = mean(Wcomp<z(i));
end

FW = smoothing(FW);
FWcomp = smoothing(FWcomp);


%%%% cut of 
% z = z(5:(end-4));
% FW = FW(5:(end-4));
% FWcomp = FWcomp(5:(end-4));

figure
plot(z,abs(FW-normcdf(z)), 'r-', 'LineWidth', 2);
xlim([-0.5 0.5] )
hold on
plot(z,abs(FWcomp-normcdf(z)), 'g-', 'LineWidth', 2);

    title('Empirical Distribution vs Normal Distribution')
    xlabel('z')
    ylabel({'$ \bar{F}_n(z)-\Phi(z)$'},'Interpreter','latex')
    legend("Our method", "Former method")
    ax = gca;
ax.FontSize = 18;
%plot(z,normcdf(z), 'k-', 'LineWidth', 2);


% subplot(1,2,1);
% histogram(normcdf(W(:,1)),'Normalization','probability');
% title('p-value of $W_T$','interpreter','latex');
% subplot(1,2,2);
% histogram(W(:,1).*W(:,2),-3:0.3:3,'Normalization','probability')
% title('Symmetricity of $W_T^{(1)}\cdot W_T^{(2)}$','interpreter','latex');


function z=smoothing(a)
lz = length(a);
z = zeros(1,lz); 
K =3;
for i = 1:lz
    idxL = max(i-K,1);
    idxU = min(i+K,lz);

    z(i)=mean(a(idxL:idxU));
    
    % if i==1
    %     z(i)= (a(i)+a(i+1))/2;
    % elseif i == lz
    %     z(i)= (a(i)+a(i-1))/2;
    % else
    %     z(i)= (a(i+1)+a(i)+a(i-1))/3;
    % end

end

end
