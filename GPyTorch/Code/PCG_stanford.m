%Standard preconditioned conjugate gradients following Stanford University


A=randn(3);
A=A'*A;
B=0.01*eye(3);
mvm_A=[6 3 0;
3 6 -6;
0 -6 11;];
b=[1;3;2];
N=4;        %times of iteration
e=0.00001;  %tolerance


%Initialization
pro=zeros(1,N);
x=zeros(3,1);
r=b;
pro(1)=norm(r,2);
for k=2:N
    if sqrt(pro(k-1))<=e*norm(b,1);
        break
    end
    if k==2
        p=r;
    else
        p=r+(pro(k-1)./pro(k-2))*p;
    end
    w=mvm_A*p;
    alpha=pro(k-1)./(p'*w);
    x=x+alpha*p;
    r=r-alpha*w;
    pro(k)=norm(r,2);
end