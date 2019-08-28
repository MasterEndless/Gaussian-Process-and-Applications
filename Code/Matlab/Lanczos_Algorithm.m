% Lanczos Algorithm
% Create a random symmetric matrix
D=200;
for i=1:D,
    for j=1:i,
        A(i,j)=rand;
        A(j,i)=A(i,j);
    end
end
% Iteration with j=0
r=zeros(D,D+1);
q=zeros(D,D+1);
alpha=zeros(D+1,1);
beta=zeros(D+1,1);

I=eye(D);
r(:,1)=rand(D,1);
beta(1)=sqrt(r(:,1)'*r(:,1));

%program start
for j=1:201
    if beta(j)~=0
        q(:,j+1)=r(:,j)./beta(j);
        alpha(j+1)=q(:,j+1)'*A*q(:,j+1);
        r(:,j+1)=(A-alpha(j+1)*I)*q(:,j+1)-beta(j)*q(:,j);
        beta(j+1)=sqrt(norm(r(:,j+1),2));
    end
end
Q_pre=zeros(D,1);
for i=2:j
    Q_pre=[Q_pre;q(j)];
end
Q=Q_pre(:,2:end);

