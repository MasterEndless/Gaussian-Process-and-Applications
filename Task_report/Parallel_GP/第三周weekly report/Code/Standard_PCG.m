%Standard preconditioned conjugate gradients
function [ c ] = Standard_PCG( mvm_A,b, M )
n=size(mvm_A,1);
T=50;        %number of iteration
e=0.0001;   %tolerance


%initialization
u=zeros(n,T);
r=zeros(n,T);
z=zeros(n,T);
d=zeros(n,T);
v=zeros(n,T);
alpha=zeros(1,T);
beta=zeros(1,T);
truth=inv(mvm_A)*b;

%main program
r(:,1)=b-mvm_A*u(:,1);          %error fixed
z(:,1)=M*r(:,1);            %changes to PCG
d(:,1)=z(:,1);
E=zeros(1,T);

for j=2:T
    v(:,j)=mvm_A*d(:,j-1);
    alpha(j)=(z(:,j-1)'*r(:,j-1))./(d(:,j-1)'*v(:,j));
    u(:,j)=u(:,j-1)+alpha(j)*d(:,j-1);
    r(:,j)=r(:,j-1)-alpha(j)*v(:,j);
    if norm(r(:,j),2)<e
        break
    end
    z(:,j)=M*r(:,j);        %changes to PCG
    beta(j)=(z(:,j)'*r(:,j))./(z(:,j-1)'*r(:,j-1));     %error fixed
    d(:,j)=z(:,j)+beta(j)*d(:,j-1);                 %error fixed
    E(j)=norm(truth-u(:,j),1);
end
E(1)=E(2);
n=1:T;
figure(1);
plot(n,E);xlabel('N');ylabel('Error');title('Error of Preconditioned PCG during each iteration time');
c=E;



    
        
    

