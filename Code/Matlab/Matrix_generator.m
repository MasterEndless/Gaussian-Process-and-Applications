%Function to generate [y z1 z2 ... zt]
function [B] = Matrix_generator(K,y,t)
% t represents how much z_i you want to produce
row=size(K,1);              %dimension of matrix
Z=-1 + 2.*rand([row t]);

for k = 1:row
    for i=1:t
        if Z(k,i)>0
            Z(k,i)=1;
        else
            Z(k,i)=-1;
        end
    end
end

B=[y Z];