clear;
addpath('./util/') % use our utility functions

d1 = 1000;
d2 = 1000;
r = 3;
signalr = 1000;
n= 20*r*d1;



rng(1000);
A=randn(d1,r);
B=randn(r,d2);
[U,~,~]=svds(A,r);
X = U;
[~,~,V]=svds(B,r);
Y =V;

M= X*diag(repelem(signalr,r) )*Y.';


seed = 1111;
rng(seed);



%%%% resample 1
% save('VarD.mat',"Data")
% VD = load('VarD.mat');
% Data = VD.Data;

    % h1 = 10;
    % h2 = 20;
    % T1 = zeros(h1*h2,2); % tests
    % for i= 1:h1
    %     for j = 1:h2
    %         T1( i+ (h1)*(j-1),: ) = [i+10,j+10];
    %     end 
    % end
    h1 = 20;
     h2 = 20;

    T1 = zeros(h1*h2,2); % tests

    for i = 1: h1*h2
        T1(i,:) = [i+10,i+10];
    end

      %    Sigma = ProjT(U,V,T1,"single");
      % save('S_single.mat',"Sigma");
    Sdata = load('S_single.mat');
    Sigma = Sdata.Sigma;
    [A,S,B] = svd(Sigma);
    Sdiag  = diag(S);
    Sdiag(Sdiag<0)=0;
    Splus = diag(Sdiag);
    Sigma = A * Splus * B';


    % assign Null vs Non-null
    p=0.2;
    mu = 0.3;

    seed = 1000;
    rng(seed);
    sigHypo = 2*(rand(h1*h2,1)>0.5)-1; % randi([0,1], h1*h2, 1)*2 - 1;
    Hypo= (1+0.5*mu-mu*rand(h1*h2,1)).*sigHypo.*binornd(1,p,h1*h2,1);
    
       Hypok = Hypo( 1 :h1*h2);
        Tk = [T1, Hypok];
        truedis = Hypok~=0;
        faldis = Hypok==0;
    % 
    % save('VarT.mat',"Hypo");
    % VarT = load('VarT.mat');
    % Hypo = VarT.Hypo;
    Rdaveg = 96;

    X = inv(real(sqrtm(Sigma)) );
    Scov = inv(diag(sqrt(diag(Sigma)) ))*Sigma*inv(diag( sqrt(diag(Sigma)) ));

    rho = sum(abs(Scov)>0.2,"all")/(h1*h2)^2
    



Data = CreateData(M, n, 2*Rdaveg, "gaussian");

signalT = 0.4:0.02:0.6;
epoch   = length(signalT);

alpha   = 0.05;   

nMethods = 6;

chunkSize = 10;
nChunks   = ceil(Rdaveg / chunkSize);

% total accumulators over all replications
FDR_tot = zeros(nMethods, epoch);
POW_tot = zeros(nMethods, epoch);

for c = 1:nChunks
    % indices of k in this chunk
    k_start = (c-1)*chunkSize + 1;
    k_end   = min(c*chunkSize, Rdaveg);
    k_list  = k_start:k_end;
    nk      = numel(k_list);

    % per-chunk storage (no 3D numeric tensors, just cells)
    FDR_cell = cell(nk, 1);
    POW_cell = cell(nk, 1);

    parfor idx = 1:nk
        k = k_list(idx);

        

        % local accumulators for this k
        FDR_k = zeros(nMethods, epoch);
        POW_k = zeros(nMethods, epoch);

        % if you want different Hypo per round, use sliced version:
        % Hypok = Hypo( ((k-1)*h1*h2+1) : k*h1*h2 );
        Hypok = Hypo(1:h1*h2);   % your current choice

        Tk = [T1, Hypok];

        truedis = Hypok ~= 0;
        faldis  = Hypok == 0;

        D1   = Data(k,:,:) ;
        D2   = Data(k+Rdaveg,:,:);
        Dall = [D1; D2];
              
        W1all = Feature(M, D1,   Tk, r, signalT, "entry","single");
        W2all = Feature(M, D2,   Tk, r, signalT, "entry","single");
        W3all = Feature(M, Dall, Tk, r, signalT, "entry","double");

        denom_true = sum(truedis);
        if denom_true == 0
            denom_true_eff = 1;    % avoids div-by-zero; POW will be 0
        else
            denom_true_eff = denom_true;
        end

        for sig = 1:epoch


            W1 = W1all(:, sig);
            W2 = W2all(:, sig);
            W3 = W3all(:, sig);

            % SDA score
            Wsda = SDA(W1, W2, X, Sigma, Tk,"single");

            % drop non-positive
           drop = 0;
           if drop
                Z      = (W1 < 0) & (W2 < 0);
                W1(Z)  = 0;
                W2(Z)  = 0;
           end

            Wrank = W1 .* W2;
            Wcomp = sign(W1).*sign(W2).*min(abs(W1),abs(W2));
            Wadd  = sign(W1).*sign(W2).*(abs(W1)+abs(W2));

            % 3 methods based on W1,W2
            res1 = Discovery(Wrank, alpha);
            res2 = Discovery(Wcomp, alpha);
            res3 = Discovery(Wadd, alpha);
            reslocalBH = local_fdr_procedure(W3,alpha,100,[0.125 0.25 0.375 0.5 0.625 0.75 0.775]);

            res6 = reslocalBH;

            % SDA 

            res4      = Discovery(Wsda, alpha);
  
            
            % BHq on W3
            BH   = 2*(1 - normcdf(abs(W3)));
            res5 = BHq(BH, alpha);

            % % factor-adjusted FDP
            % [disc, ~, ~, ~, ~] = fanhan_factor_adjust_discoveries(W3, Sigma, alpha);
            % res6 = disc;
            % 
            % % Asymptotic e-BH
            % evalue  =  exp(W3*signalT(sig)-1/2* signalT(sig)^2);
            % res7 = BHq(1./evalue, alpha);

            % ----- update local FDR/POW for this k -----
            FDR_k(1,sig) = FDR_k(1,sig) + sum(res1(faldis) == 1) / max(sum(res1 == 1), 1);
            POW_k(1,sig) = POW_k(1,sig) + sum(res1(truedis) == 1) / denom_true_eff;

            FDR_k(2,sig) = FDR_k(2,sig) + sum(res2(faldis) == 1) / max(sum(res2 == 1), 1);
            POW_k(2,sig) = POW_k(2,sig) + sum(res2(truedis) == 1) / denom_true_eff;

            FDR_k(3,sig) = FDR_k(3,sig) + sum(res3(faldis) == 1) / max(sum(res3 == 1), 1);
            POW_k(3,sig) = POW_k(3,sig) + sum(res3(truedis) == 1) / denom_true_eff;

            FDR_k(4,sig) = FDR_k(4,sig) + sum(res4(faldis) == 1) / max(sum(res4 == 1), 1);
            POW_k(4,sig) = POW_k(4,sig) + sum(res4(truedis) == 1) / denom_true_eff;

            FDR_k(5,sig) = FDR_k(5,sig) + sum(res5(faldis) == 1) / max(sum(res5 == 1), 1);
            POW_k(5,sig) = POW_k(5,sig) + sum(res5(truedis) == 1) / denom_true_eff;

            FDR_k(6,sig) = FDR_k(6,sig) + sum(res6(faldis) == 1) / max(sum(res6 == 1), 1);
            POW_k(6,sig) = POW_k(6,sig) + sum(res6(truedis) == 1) / denom_true_eff;

           
        end

        % stash this replication's result in cells
        FDR_cell{idx} = FDR_k;
        POW_cell{idx} = POW_k;
    end

    % combine this chunk's results on the client
    for idx = 1:nk
        FDR_tot = FDR_tot + FDR_cell{idx};
        POW_tot = POW_tot + POW_cell{idx};
    end

    fprintf('finished chunk %d / %d\n', c, nChunks);
end

% final averages over all Rdaveg replicates
FDR = FDR_tot / Rdaveg;
POW = POW_tot / Rdaveg;


%%
    save("local_FDR_res.mat","POW","FDR");
    localres = load("local_FDR_res.mat");
    POW = localres.POW;
    FDR = localres.FDR;
   figure;
   subplot(1,2,1)

    plot(signalT,FDR(1,:),'g-o','LineWidth',1)
    hold on

    plot( signalT,FDR(2,:),'b-*','LineWidth',1)
    
    plot(    signalT,FDR(3,:),'r-^','LineWidth',1) 
    plot(signalT,FDR(4,:),'c-square','LineWidth',1)
    plot (signalT,FDR(5,:),'k-+','LineWidth',1)
    plot (signalT,FDR(6,:),'m-x','LineWidth',1)
   plot (signalT,alpha* ones(1,length(signalT)),'b--','LineWidth',1.5)
    % title('FDR Control')
    xlabel('signal')
    ylabel('FDP')
    legend("multiply", "min", "add", "SDA","BH","local FDR")
    xlim([min(signalT),max(signalT)])
    ax = gca;
ax.FontSize = 18;
       subplot(1,2,2)
        plot(signalT,POW(1,:),'g-o','LineWidth',1)
    hold on
    
    plot( signalT,POW(2,:),'b-*','LineWidth',1)
    
    plot(    signalT,POW(3,:),'r-^','LineWidth',1) 
    plot(signalT,POW(4,:),'c-square','LineWidth',1)
    plot (signalT,POW(5,:),'k-+','LineWidth',1)
    plot (signalT,POW(6,:),'m-x','LineWidth',1)

    %title('Power Control')
    xlabel('signal')
    ylabel('Power')
     xlim([min(signalT),max(signalT)])
    legend("multiply", "min", "add", "SDA","BH","local FDR")
ax = gca;
ax.FontSize = 18;
sgtitle('FDP and Power Performance', 'FontSize', 22, 'FontWeight', 'bold');