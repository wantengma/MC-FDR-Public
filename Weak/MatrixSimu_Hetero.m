% N = 12; % Set the desired number of workers
% poolobj = parpool('local', N);
% numWorkers = poolobj.NumWorkers; % Get the number of assigned workers
% disp(['Using ', num2str(numWorkers), ' workers.']);

clear;
addpath('./util/') % use our utility functions

d1 = 1000;
d2 = 1000;
r = 3;
signalr = 1000;
n= 50*r*d1;
alpha = 0.1;


rng(1000);
A=randn(d1,r);
B=randn(r,d2);
[U,~,~]=svds(A,r);
X = U;
[~,~,V]=svds(B,r);
Y =V;

M= X*diag(repelem(signalr,r) )*Y.';

%Null Hypothesis M=m


%% 

TestCase = 1; %choose which test we want to do: 
% 0-> block tests
% 1-> LASSO SDA
% 2-> entry comparison

if TestCase == 0 % do block tests:
    %%%% 
    %%%% 
    %%%% 
    %%%% block tests:
    
    
seed = 1111;
rng(seed);



%%%% resample 1
% save('VarD.mat',"Data")
% VD = load('VarD.mat');
% Data = VD.Data;

    h1 = 100;
    h2 = 100;
    T1 = zeros(h1*h2,2); % tests
    for i= 1:h1
        for j = 1:h2
            T1( i+ (h1)*(j-1),: ) = [i+10,j+10];
        end 
    end
    
    % assign Null vs Non-null
    p=0.2;
    mu = 0.3;

    seed = 1000;
    rng(seed);
    sigHypo =2*(rand(h1*h2,1)>0.5)-1; % randi([0,1], h1*h2, 1)*2 - 1;
    Hypo= (1+0.5*mu-mu*rand(h1*h2,1)).*sigHypo.*binornd(1,p,h1*h2,1);
    
       Hypok = Hypo( 1 :h1*h2);
        Tk = [T1, Hypok];
        truedis = Hypok~=0;
        faldis = Hypok==0;
    % 
    % save('VarT.mat',"Hypo");
    % VarT = load('VarT.mat');
    % Hypo = VarT.Hypo;
    Rdaveg = 12;

Data = CreateData(M, n, 2*Rdaveg, "hetero");

signalT = 0.4:0.2:1.4;
epoch   = length(signalT);

alpha   = 0.1;   

% --- true/false discovery masks and Tk assumed prepared earlier ---
% Hypok, truedis, faldis, Tk
% (same as in your original code)

% sanity: how many (D1,D2) replicate pairs do we really have?
nDataReps = size(Data,1) / 2;
if Rdaveg > nDataReps
    error('Rdaveg (%d) > number of replicate pairs in Data (%d).', ...
          Rdaveg, nDataReps);
end

% number of methods actually computed here (res1, res2, res3, BHq)
nMethods = 4;

% global accumulators
FDR_tot = zeros(nMethods, epoch);
POW_tot = zeros(nMethods, epoch);

% ---------- chunked parfor settings ----------
chunkSize = 12;                        % you can tune this
nChunks   = ceil(Rdaveg / chunkSize);

for c = 1:nChunks
    % indices of k in this chunk
    k_start = (c-1)*chunkSize + 1;
    k_end   = min(c*chunkSize, Rdaveg);
    k_list  = k_start:k_end;
    nk      = numel(k_list);

    % per-chunk storage (no large 3D numeric tensors)
    FDR_cell = cell(nk, 1);
    POW_cell = cell(nk, 1);

    parfor idx = 1:nk
        k = k_list(idx);

        % local accumulators for this replication
        FDR_k = zeros(nMethods, epoch);
        POW_k = zeros(nMethods, epoch);

        % data for this k
        D1   = Data(k,             :, :);
        D2   = Data(k + nDataReps, :, :);
        Dall = [D1; D2];

        % W1/W2/W3 are T x epoch matrices
        W1mat = Feature(M, D1,   Tk, r, signalT, "entry");
        W2mat = Feature(M, D2,   Tk, r, signalT, "entry");
        W3mat = Feature(M, Dall, Tk, r, signalT, "entry", "double");

        denom_true = sum(truedis);
        if denom_true == 0
            denom_true_eff = 1;  % avoids division by 0; POW will be 0
        else
            denom_true_eff = denom_true;
        end

        for sig = 1:epoch
            W1 = W1mat(:, sig);
            W2 = W2mat(:, sig);
            W3 = W3mat(:, sig);

            Wrank = W1 .* W2;

            Wcomp = sign(W1).*sign(W2).*min(abs(W1),abs(W2));
            Wadd  = sign(W1).*sign(W2).*(abs(W1)+abs(W2));

            % three composite methods
            res1 = Discovery(Wrank, alpha);
            res2 = Discovery(Wcomp, alpha);
            res3 = Discovery(Wadd, alpha);

           % drop non-positive
           drop = 1;
           if drop
                Z      = (W1 < 0) & (W2 < 0);
                W1(Z)  = 0;
                W2(Z)  = 0;
           end

            % BHq on W3
            BH   = 2*(1 - normcdf(abs(W3)));     %1 - normcdf(W3);  2*(1 - normcdf(abs(W3)));   
            res4 = BHq(BH, alpha);

            % FDR/POW contributions for this replication / signalT(sig)
            FDR_k(1,sig) = sum(res1(faldis) == 1) / max(sum(res1 == 1), 1);
            POW_k(1,sig) = sum(res1(truedis) == 1) / denom_true_eff;

            FDR_k(2,sig) = sum(res2(faldis) == 1) / max(sum(res2 == 1), 1);
            POW_k(2,sig) = sum(res2(truedis) == 1) / denom_true_eff;

            FDR_k(3,sig) = sum(res3(faldis) == 1) / max(sum(res3 == 1), 1);
            POW_k(3,sig) = sum(res3(truedis) == 1) / denom_true_eff;

            FDR_k(4,sig) = sum(res4(faldis) == 1) / max(sum(res4 == 1), 1);
            POW_k(4,sig) = sum(res4(truedis) == 1) / denom_true_eff;
        end

        % stash this replication's result
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
  %  save('results_weak_d=1000.mat', 'FDR', 'POW')
    figure;

    subplot(1,2,1)
    plot(signalT,FDR(1,:),'g-o', 'LineWidth', 1)
    hold on
    plot(signalT,FDR(2,:),'b-*', 'LineWidth', 1)
    plot(signalT,FDR(3,:),'r-^', 'LineWidth', 1)
    plot(signalT,FDR(4,:),'k-+', 'LineWidth', 1)
    xlim([min(signalT),max(signalT)])
    ylim([0,0.2])
    xlabel('signal')
    ylabel('FDP')
    legend("multiply", "min", "add","BH",'Location','northeast')
     ax = gca;
 ax.FontSize = 18;

    subplot(1,2,2)
    plot(signalT,POW(1,:),'g-o', 'LineWidth', 1)
    hold on
    plot(signalT,POW(2,:),'b-*', 'LineWidth', 1)
    plot(signalT,POW(3,:),'r-^', 'LineWidth', 1)
    plot(signalT,POW(4,:),'k-+', 'LineWidth', 1)
  
    xlabel('signal')
    ylabel('Power')
    legend("multiply", "min", "add","BH",'Location','northeast')

     ax = gca;
 ax.FontSize = 18;

    sgtitle('FDP and Power Performance', 'FontSize', 22, 'FontWeight', 'bold');
    saveas(gcf,'d=1000')



   % 
   % %%%% 
   %  %%%%     
   % 
   %  plot(signalT,FDR(1,:),'g-o',signalT,FDR(2,:),'b-o',signalT,FDR(3,:),'r-o',signalT,FDR(4,:),'m-o', signalT,FDR(5,:),'k-o')
   %  title('FDR Control')
   %  xlabel('signal')
   %  ylabel('FDP')
   %  legend("multiply", "min", "add","BHq")
   % 
   %  figure;
   %  plot(signalT,POW(1,:),'g-o',signalT,POW(2,:),'b-o',signalT,POW(3,:),'r-o',signalT,POW(4,:),'m-o',signalT,POW(5,:),'k-o')
   %  title('Power Control')
   %  xlabel('signal')
   %  ylabel('Power')
   %  legend("multiply", "min","add","BHq")



elseif TestCase == 1 % do lasso SDA tests:




Rdaveg = 48; % 48, 96, 24
seed   = 111111;
rng(seed);

% Must be reruned after changing Rdaveg
Data = CreateData(M, n, 2*Rdaveg, "hetero");




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
        [U,~,V]=svds(M,r);
     %     Sigma = ProjT(U,V,T1);
     % save('S_rowdiff.mat',"Sigma");
    Sig = load('S_rowdiff.mat');
    Sigma = Sig.Sigma;

    % randomly assign Null vs Non-null 
    p=0.2;
    mu = 0.3;

    groundT = 1;
    if groundT
    
    
    sigHypo = randi([0,1], Rdaveg*h1*h2, 1)*2 - 1;
    Hypo= (1+0.5*mu-mu*rand(Rdaveg*h1*h2,1)).*sigHypo.*binornd(1,p,Rdaveg*h1*h2,1);
    
    
     save('VarT2.mat',"Hypo");
     VarT = load('VarT2.mat');
     Hypo = VarT.Hypo;



    end
    X = inv(sqrtm(Sigma) );
    Scov = inv(diag(sqrt(diag(Sigma)) ))*Sigma*inv(diag( sqrt(diag(Sigma)) ));

    rho = sum(abs(Scov)>0.2,"all")/(h1*h2)^2
    


signalT   = 0.2:0.1:1.0;
epoch     = length(signalT);

% parpool('local',12);   
% choose a chunk size that you know is stable (e.g. 12 or 16)
chunkSize = 12;
nChunks   = ceil(Rdaveg / chunkSize);

% total accumulators over all replications
FDR_tot = zeros(7, epoch);
POW_tot = zeros(7, epoch);

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

        alpha0 = 0.1;

        % local accumulators for this k
        FDR_k = zeros(7, epoch);
        POW_k = zeros(7, epoch);

        % if you want different Hypo per round, use sliced version:
        % Hypok = Hypo( ((k-1)*h1*h2+1) : k*h1*h2 );
        Hypok = Hypo(1:h1*h2);   % your current choice

        Tk = [T1, Hypok];

        truedis = Hypok ~= 0;
        faldis  = Hypok == 0;

        D1   = Data(k,:,:) ;
        D2   = Data(k+Rdaveg,:,:);
        Dall = [D1; D2];
              
        W1all = FeatureHetero(M, D1,   Tk, r, signalT, "two-diff","single");
        W2all = FeatureHetero(M, D2,   Tk, r, signalT, "two-diff","single");
        W3all = FeatureHetero(M, Dall, Tk, r, signalT, "two-diff","double");

        denom_true = sum(truedis);
        if denom_true == 0
            denom_true_eff = 1;    % avoids div-by-zero; POW will be 0
        else
            denom_true_eff = denom_true;
        end

        for sig = 1:epoch
            alpha = alpha0;

            W1 = W1all(:, sig);
            W2 = W2all(:, sig);
            W3 = W3all(:, sig);

            % SDA score
            Wsda = SDA(W1, W2, X, Sigma, Tk);

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

            % SDA 

            res4      = Discovery(Wsda, alpha);

            
            % BHq on W3
            BH   = 2*(1 - normcdf(abs(W3)));
            res5 = BHq(BH, alpha);

            % factor-adjusted FDP
            [disc, ~, ~, ~, ~] = fanhan_factor_adjust_discoveries(W3, Sigma, alpha);
            res6 = disc;

            % Asymptotic e-BH
            evalue  =  exp(W3*signalT(sig)-1/2* signalT(sig)^2);
            res7 = BHq(1./evalue, alpha);

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

            FDR_k(7,sig) = FDR_k(7,sig) + sum(res7(faldis) == 1) / max(sum(res7 == 1), 1);
            POW_k(7,sig) = POW_k(7,sig) + sum(res7(truedis) == 1) / denom_true_eff;
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
  % save('results_weak_diff_d=1000.mat', 'FDR', 'POW')
   figure;
   subplot(1,2,1)
    plot(signalT,FDR(1,:),'g-o','LineWidth',1)
    hold on

    plot( signalT,FDR(2,:),'b-*','LineWidth',1)
    
    plot(    signalT,FDR(3,:),'r-^','LineWidth',1) 
    plot(signalT,FDR(4,:),'c-square','LineWidth',1)
    plot (signalT,FDR(5,:),'k-+','LineWidth',1)
    plot (signalT,FDR(6,:),'m-x','LineWidth',1)
    plot (signalT,FDR(7,:),'k-diamond','LineWidth',1)
    % title('FDR Control')
    xlabel('signal')
    ylabel('FDP')
    legend("multiply", "min", "add", "SDA","BH","FPA","eBH")
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
    plot (signalT,POW(7,:),'k-diamond','LineWidth',1)
    %title('Power Control')
    xlabel('signal')
    ylabel('Power')
    legend("multiply", "min", "add", "SDA","BH","FPA","eBH")
ax = gca;
ax.FontSize = 18;
sgtitle('FDP and Power Performance', 'FontSize', 22, 'FontWeight', 'bold');
end




