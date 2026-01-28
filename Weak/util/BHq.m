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