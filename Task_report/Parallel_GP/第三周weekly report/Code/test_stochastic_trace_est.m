% stochastic trace estimation
n=30;              %dimension of matrix
N=200;                %sampling times
A = rand(n);
B = tril(A,-1)+triu(A',0);
x=-1 + 2.*rand([n N]);
for k = 1:n
    for i=1:N
        if x(k,i)>0
            x(k,i)=1;
        else
            x(k,i)=-1;
        end
    end
end
s=0;
count=1;
tr_err=zeros(1,N);
tr=trace(B);
for i = 1:N
    s=s+x(:,i)'*B*x(:,i);
    tr_err(i)=s/count-tr;
    count=count+1;
end
figure(1);
n=1:N;
plot(n,tr_err);xlabel('N');ylabel('Error');title('The error during stochastic trace estimation');




