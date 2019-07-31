%Implementation of Pivited Cholesky Composition
A=[6 3 0;
3 6 -6;
0 -6 11];    %semi-positive matrix

e=0.1;            %Error
n=size(A,1);

m=1;
d=diag(A);
error=norm(d,1);
v=randperm(n);
Pi = sort(v);
l=zeros(n,n);   %triangular matrix we want 



while error>e
    [argvalue, i] = max(d(Pi(m:end)));
    a=Pi(m);
    Pi(m)=Pi(i);
    Pi(i)=a;
    l(m,Pi(m))=sqrt(d(Pi(m)));
    for i = (m+1:n)
        S=0;
        for j =(1:m-1)
            l(j,Pi(m))
            l(j,Pi(i))
            S=S+l(j,Pi(m)).*l(j,Pi(i));
        end
        l(m,Pi(i))=(A(Pi(m),Pi(i))-S)./l(m,Pi(m));
        d(Pi(i))=d(Pi(i))-l(m,Pi(m)).*l(m,Pi(i));
    end
    %compute error
    error_s=0;
    for i = (m+1:n)
        error_s=error_s+d(Pi(i));
    end
    error=error_s;
    m=m+1;      
end
S_Am=zeros(3,3);
for i=(1:m-1)
    S_Am=S_Am+l(:,i)*l(:,i)';
end
Am=S_Am;


    




