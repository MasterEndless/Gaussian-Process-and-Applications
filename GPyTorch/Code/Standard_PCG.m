%Standard preconditioned conjugate gradients
mvm_A=[6 3 0;
3 6 -6;
0 -6 11;];
b=[4;7;9];
P_1=eye(3);
T=5;        %number of iteration
e=0.0001;   %tolerance

%initialization
u=zeros(3,T);
r=zeros(3,T);
z=zeros(3,T);
d=zeros(3,T);
v=zeros(3,T);
alpha=zeros(1,T);
beta=zeros(1,T);


%main program
r(:,1)=b-mvm_A*u(:,1);          %error fixed
z(:,1)=P_1*r(:,1);
d(:,1)=z(:,1);

for j=2:T
    v(:,j)=mvm_A*d(:,j-1);
    alpha(j)=(r(:,j-1)'*z(:,j-1))./(d(:,j-1)'*v(:,j));
    u(:,j)=u(:,j-1)+alpha(j)*d(:,j-1);
    r(:,j)=r(:,j-1)-alpha(j)*v(:,j);
    if norm(r(:,j),2)<e
        break
    end
    z(:,j)=P_1*r(:,j);
    beta(j)=(z(:,j)'*z(:,j))./(z(:,j-1)'*z(:,j-1));
    d(:,j)=z(:,j)+beta(j)*d(:,j-1);                 %error fixed
end
display(u(:,j))
    
        
    

