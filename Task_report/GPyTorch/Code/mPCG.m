%Modified preconditioned conjugate gradients
function [ Stor,T ] = mPCG( mmm_A,B,M)
row=size(mmm_A,1);
t=row;        %number of iteration
col=size(B,2);

U=zeros(row,col);
R=B-mmm_A*U;
Z=M*R;
D=Z;

e=0.00001;       % allowed error
flag=0;
T=zeros(t,t,col);

alpha=zeros(t,1);
beta=zeros(t,1);
for j=1:t
    V=mmm_A*D;
    if j>1
        alpha_old=alpha;       %store old alpha
        beta_old=beta;
    end
    alpha=(dot(R,Z)./dot(D,V))';
    display(alpha)
    U= (U'+diag(alpha)*D')';
    Z_old=dot(Z,R);
    R= (R'-diag(alpha)*V')';
    norm_matrix=zeros(1,col);
    for i=1:col
        norm_matrix(i)=norm(R(:,i),2);
    end
    if any(any(norm_matrix<e))
        Stor=U;
        flag=1;
    end
    Z=M*R;
    beta=(dot(Z,R)./Z_old)';
    display(beta)
    D=(Z'+diag(beta)*D')';
    if flag==0
        Stor=U;
    end
    
    %Implementation of Lanczos_Algorithm
    for i = 1:t
        if j==1
            T(j,j,i)=1./alpha(i);
        else
            T(j,j,i)=1./alpha(i)+beta_old(i)./alpha_old(i);
            T(j-1,j,i)=sqrt(beta_old(i))./alpha(i);
            T(j,j-1,i)=T(j-1,j,i);
        end
    end
    
    
end
%T=Lanczos_Algorithm(Alpha,Beta);
        


