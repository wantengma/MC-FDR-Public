clear;
addpath('./util/') % use our utility functions


d1 = 400;
d2 = 400;
r = 3;
signalr = 400;
n= 3000;
alpha =0.1;



M = CreateLowM(d1,d2,signalr,r);

%%%%save('VarM_noise.mat',"M");
VarM = load('VarM_noise.mat');

M = VarM.M;



seed = 111111;
rng(seed);

%Null Hypothesis M=m

   
Rdaveg = 96;
exp_type = "student"; % "student"
    Data = CreateData(M,n,2*Rdaveg,exp_type);



    %%%% 
    %%%% 
    h1 = d1-1;
    h2 = 4;
    T1 = zeros(h1*h2,4); % tests: ->coordinate 1- coordinate 2
    for i= 1:h1
        for j = 1:h2
            T1( i+ (h1)*(j-1),1:2 ) = [1,1];
            T1( i+ (h1)*(j-1),3:4 ) = [i+1,j];
        end 
    end
    
    % randomly assign Null vs Non-null
    p=0.2;
    mu = 0.3;
    seed = 1111111;
    rng(seed);
    sigHypo = 1; %randi([0,1], Rdaveg*h1*h2, 1)*2 - 1;
    Hypo= (1+0.5*mu-mu*rand(Rdaveg*h1*h2,1)).*sigHypo.*binornd(1,p,Rdaveg*h1*h2,1);
    
    
     % save('VarT_noise.mat',"Hypo");
     % VarT = load('VarT_noise.mat');
     % Hypo = VarT.Hypo;

    
    

     [U,~,V] = svds(M,r);
% Sigma = ProjT(U,V,T1);
%    save('S_noise.mat',"Sigma");
    Sig = load('S_noise.mat');
    
    Sigma = Sig.Sigma; % SS= diag(1./sqrt(diag(Sigma)) )*Sigma*diag(1./sqrt(diag(Sigma)) );  sum(abs(SS)>0.2,'all')/size(SS,1)^2
    q = size(Sigma,1); 
    [Us,S] = schur(Sigma);%+0.0005*eye(q)
    X = Us*  (pinv(sqrt(S),0.0001) +0.0001*eye(q))*Us.';


      
    %[U,V]= Subspace(M,D1,r); % estimation of singular subspace
signalT = 1.5:0.1:2.5;
epoch   = length(signalT);

   
nMethods = 7;        

% sanity: how many (D1,D2) pairs in Data?
nDataReps = size(Data,1) / 2;
if Rdaveg > nDataReps
    error('Rdaveg (%d) > number of replicate pairs in Data (%d).', ...
          Rdaveg, nDataReps);
end

% ---------- chunk settings ----------
chunkSize = 12;                     
nChunks   = ceil(Rdaveg / chunkSize);

% global accumulators
FDR_tot = zeros(nMethods, epoch);
POW_tot = zeros(nMethods, epoch);

for c = 1:nChunks
    % indices of k in this chunk
    k_start = (c-1)*chunkSize + 1;
    k_end   = min(c*chunkSize, Rdaveg);
    k_list  = k_start:k_end;
    nk      = numel(k_list);

    % per-chunk storage (avoid large 3D arrays)
    FDR_cell = cell(nk, 1);
    POW_cell = cell(nk, 1);

    parfor idx = 1:nk
        alpha =0.1;
        k = k_list(idx);

        % local accumulators for this replication
        FDR_k = zeros(nMethods, epoch);
        POW_k = zeros(nMethods, epoch);

        % same Hypo/T1/Tk for all k (as in your code)
        Hypok   = Hypo(1:h1*h2);   % or use sliced version if desired
        Tk_local = [T1, Hypok];    % avoid broadcast confusion

        truedis = Hypok ~= 0;
        faldis  = Hypok == 0;

        % data for this k
        D1   = Data(k,             :, :);
        D2   = Data(k + nDataReps, :, :);
        Dall = [D1; D2];

        % features: T x epoch
        W1mat = Feature(M, D1,   Tk_local, r, signalT, "two-diff","single");
        W2mat = Feature(M, D2,   Tk_local, r, signalT, "two-diff","single");
        W3mat = Feature(M, Dall,   Tk_local, r, signalT, "two-diff","double");
        denom_true = sum(truedis);
        if denom_true == 0
            denom_true_eff = 1;  % avoids division-by-zero; POW becomes 0
        else
            denom_true_eff = denom_true;
        end

        for sig = 1:epoch
            W1 = W1mat(:, sig);
            W2 = W2mat(:, sig);
            W3 = W3mat(:, sig);

            % SDA score
            [Wsda, ~] = SDA(W1, W2, X, Sigma, Tk_local);

            % optional dropping 
            Drop = 0;
            if Drop == 1
                Z      = (W1 < 0) & (W2 < 0);
                W1(Z)  = 0;
                W2(Z)  = 0;
            end

            Wrank = W1 .* W2;

            % corW unused in selection; safe to drop if you like
            % corW = corr(W1(faldis), W2(faldis));

            Wcomp = sign(W1).*sign(W2).*min(abs(W1),abs(W2));
            Wadd  = sign(W1).*sign(W2).*(abs(W1)+abs(W2));

            % BHq selection by W3
            res1 = Discovery(Wrank, alpha);
            res2 = Discovery(Wcomp, alpha);
            res3 = Discovery(Wadd, alpha);

           
            res4 = Discovery(Wsda,  alpha);

            BH   = 2*(1 - normcdf(abs(W3)));
            res5 = BHq(BH, alpha);

            % Asymptotic e-BH
            evalue = exp(W3 * signalT(sig) - 0.5 * signalT(sig)^2);
            res7   = BHq(1./evalue, alpha);

            % factor-adjusted FDP
            [disc, ~, ~, ~, ~] = fanhan_factor_adjust_discoveries(W3, Sigma, alpha);
            res6 = disc;

            % --- accumulate local FDR/POW (methods 1–5 as in your original) ---
            FDR_k(1,sig) = FDR_k(1,sig) + ...
                sum(res1(faldis) == 1) / max(sum(res1 == 1), 1);
            POW_k(1,sig) = POW_k(1,sig) + ...
                sum(res1(truedis) == 1) / denom_true_eff;

            FDR_k(2,sig) = FDR_k(2,sig) + ...
                sum(res2(faldis) == 1) / max(sum(res2 == 1), 1);
            POW_k(2,sig) = POW_k(2,sig) + ...
                sum(res2(truedis) == 1) / denom_true_eff;

            FDR_k(3,sig) = FDR_k(3,sig) + ...
                sum(res3(faldis) == 1) / max(sum(res3 == 1), 1);
            POW_k(3,sig) = POW_k(3,sig) + ...
                sum(res3(truedis) == 1) / denom_true_eff;

            FDR_k(4,sig) = FDR_k(4,sig) + ...
                sum(res4(faldis) == 1) / max(sum(res4 == 1), 1);
            POW_k(4,sig) = POW_k(4,sig) + ...
                sum(res4(truedis) == 1) / denom_true_eff;

            FDR_k(5,sig) = FDR_k(5,sig) + ...
                sum(res5(faldis) == 1) / max(sum(res5 == 1), 1);
            POW_k(5,sig) = POW_k(5,sig) + ...
                sum(res5(truedis) == 1) / denom_true_eff;

            FDR_k(6,sig) = FDR_k(6,sig) + ...
                sum(res6(faldis) == 1) / max(sum(res6 == 1), 1);
            POW_k(6,sig) = POW_k(6,sig) + ...
                sum(res6(truedis) == 1) / denom_true_eff;

            FDR_k(7,sig) = FDR_k(7,sig) + ...
                sum(res7(faldis) == 1) / max(sum(res7 == 1), 1);
            POW_k(7,sig) = POW_k(7,sig) + ...
                sum(res7(truedis) == 1) / denom_true_eff;

            % rows 6–7 (res6/res7) are computed but not accumulated here,
            % matching your original code behavior.
        end

        FDR_cell{idx} = FDR_k;
        POW_cell{idx} = POW_k;

        fprintf('finished trial k=%d (chunk %d/%d)\n', k, c, nChunks);
    end

    % combine this chunk’s results on the client
    for idx = 1:nk
        FDR_tot = FDR_tot + FDR_cell{idx};
        POW_tot = POW_tot + POW_cell{idx};
    end

    fprintf('finished chunk %d / %d\n', c, nChunks);
end

% final averages over all Rdaveg replications
FDR = FDR_tot / Rdaveg;
POW = POW_tot / Rdaveg;



    
%% 
%% 
 save(append('results_strong_',exp_type, 'exp.mat'), 'FDR', 'POW')
  
    
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
sgtitle('FDP and Power Performance under Student-T Noise', 'FontSize', 22, 'FontWeight', 'bold');

 %   sgtitle('exponential')  %sub-Gaussian, exponential ,student-t


