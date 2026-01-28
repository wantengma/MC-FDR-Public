
function S = ProjT(U,V,Tk) %%% Tk px4 M(T1,T2)-M(T3,T4) 

    d1 = size(U,1);
    d2 = size(V,1);
    p  = size(Tk,1);

    S = zeros(p,p);

    % Precompute projectors (outside parfor)
    Uperp = eye(d1) - (U*U.');
    Vperp = eye(d2) - (V*V.');

    % Parallelize over i; each worker computes one row i (upper triangle)
    parfor i = 1:p

        % local row buffer for S(i,:)
        Si = zeros(1,p);

        % Ei depends only on i, so compute once per i
        Ei = zeros(d1,d2);
        Ei(Tk(i,1), Tk(i,2)) =  1;
        Ei(Tk(i,3), Tk(i,4)) = -1;
        Ei = Ei - ( Uperp(:,Tk(i,1))*Vperp(Tk(i,2),:) ...
                  - Uperp(:,Tk(i,3))*Vperp(Tk(i,4),:) );

        % only compute j >= i (upper triangle)
        for j = i:p
            Ej = zeros(d1,d2);
            Ej(Tk(j,1), Tk(j,2)) =  1;
            Ej(Tk(j,3), Tk(j,4)) = -1;
            Ej = Ej - ( Uperp(:,Tk(j,1))*Vperp(Tk(j,2),:) ...
                      - Uperp(:,Tk(j,3))*Vperp(Tk(j,4),:) );

            Si(j) = sum(Ei.*Ej, "all");
        end

        % write the whole row in one shot (parfor-friendly sliced assignment)
        S(i,:) = Si;
    end

    % Fill in the lower triangle by symmetry
    S = S + triu(S,1).';

end

