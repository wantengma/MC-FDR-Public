function D = CreateData(M,n,Rdaveg,NoiseType)

[d1,d2]= size(M);

D = zeros(Rdaveg,n,3);

if NoiseType == "gaussian"
for i=1:Rdaveg
   coordlinear = randsample(d1*d2,n);
   coord = zeros(d1*d2,1);
   coord(coordlinear) = 1;
   Omega = reshape(coord,d1,d2 );
   % coord1 = mod(coordlinear,d1)+1; 
   % coord2 = floor(coordlinear./d1)+1;
   [row,col,v] =  find(sparse((M+randn(d1,d2)).*Omega));
   
   D(i,:,:)= [row,col,v];
end

elseif NoiseType == "exp"
    for i=1:Rdaveg
   coordlinear = randsample(d1*d2,n);
   coord = zeros(d1*d2,1);
   coord(coordlinear) = 1;
   Omega = reshape(coord,d1,d2 );
   % coord1 = mod(coordlinear,d1)+1; 
   % coord2 = floor(coordlinear./d1)+1;
   [row,col,v] =  find (sparse((M+(exprnd(1,d1,d2)-1) ).*Omega));
   
   D(i,:,:)= [row,col,v];
    end

elseif NoiseType == "student"
   for i=1:Rdaveg
   coordlinear = randsample(d1*d2,n);
   coord = zeros(d1*d2,1);
   coord(coordlinear) = 1;
   Omega = reshape(coord,d1,d2 );
   % coord1 = mod(coordlinear,d1)+1; 
   % coord2 = floor(coordlinear./d1)+1;
   [row,col,v] =  find(sparse((M+(random('T',4,d1,d2)/sqrt(4)) ).*Omega));
   
   D(i,:,:)= [row,col,v];
    end

elseif NoiseType == "hetero"
   for i=1:Rdaveg
   coordlinear = randsample(d1*d2,n);
   coord = zeros(d1*d2,1);
   coord(coordlinear) = 1;
   Omega = reshape(coord,d1,d2 );
   % coord1 = mod(coordlinear,d1)+1; 
   % coord2 = floor(coordlinear./d1)+1;
   [row,col,v] =  find(sparse( (  floor(M)+ binornd(1,M-floor(M),d1,d2)  )  .*Omega + Omega*1e-10) );
   
   D(i,:,:)= [row,col,v];
    end
end

end