

function L=Thres(a,alpha)

alpha = 1*alpha;

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
     mu = 0.1; %0.1, L=i
      b=sort(b,"ascend");
 % alpha = min([alpha,mu*nb/na  ]);

 for i = b.'
     rej = sum(a>=i) ;
     r = (sum(a<=-i))/rej; %+1
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

