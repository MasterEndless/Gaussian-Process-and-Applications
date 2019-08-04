%Implementation of Pivited Cholesky Composition
A_=randn(10);
C=A_'*A_;
B=randn(10);
A=C+B;

e=0.01;            %Error
n=size(A,1);

m=1;
d=diag(A);
error=norm(d,1);
v=randperm(n);
Pi = sort(v);
l=zeros(n,n);   %triangular matrix we want 



while error>e
    for j=m:n
        test = d(Pi(j));
    end
    [argvalue, i]=max(test);
    a=Pi(m);
    Pi(m)=Pi(i);
    Pi(i)=a;
    l(m,Pi(m))=sqrt(d(Pi(m)));
    for i = (m+1:n)
        S=0;
        for j =(1:m-1)
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
Am=l*l';


    




