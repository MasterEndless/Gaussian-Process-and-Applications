%Modified preconditioned conjugate gradients

mmm_A=  [ -3.3704    0.2593   -3.6296;
    8.0741    0.4815    7.9259;
    5.2222    0.4444    4.7778];
B=[4 3 2;7 1 8;9 2 5];
col=size(B,2);
U=zeros(3,3);
R=B-mmm_A*U;
Z=R;
D=Z;
alpha=zeros(3,1);
beta=zeros(3,1);
t=3;        %number of iteration
e=0.0001;

for j=1:3
    V=mmm_A*D;
    alpha=(dot(R,Z)./dot(D,V))';
    U= (U'+diag(alpha)*D')';
    R= (R'-diag(alpha)*V')';
    norm_matrix=zeros(1,col);
    for i=1:col
        norm_matrix(i)=norm(R(:,i),2);
    end
    if any(any(norm_matrix<e))
        break
    end
    Z_old=dot(Z,Z);
    Z=R;
    beta=(dot(Z,Z)./Z_old)';
    D=(Z'+diag(beta)*D')';
end
display(U)
        


