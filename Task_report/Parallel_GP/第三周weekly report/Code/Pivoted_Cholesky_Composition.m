%Implementation of Pivoted Cholesky Composition
function [ G ] = Pivoted_Cholesky_Composition( A )
    n=size(A,1);
    v=randperm(n);
    Pi = sort(v);
    l=zeros(n,n);   %triangular matrix we want 
    P=zeros(n,n);
    for k=1:n
        [~, p]=max(diag(A(k:n,k:n)));
        i=p+k-1;
        
        a=A(:,k);   %exchange column
        A(:,k)= A(:,i);
        A(:,i)=a;

        b=l(:,k);       %exchange L
        l(:,k)=l(:,i);
        l(:,i)=b;

        c=A(k,:);       %exchange row
        A(k,:)=A(i,:);
        A(i,:)=c;

        a=Pi(k);        %exchange pi
        Pi(k)=Pi(i);
        Pi(i)=a;
        
        l(k,k)=sqrt(A(k,k));
        l(k,k+1:n)=l(k,k)\A(k,k+1:n);
        A(k+1:n,k+1:n)=A(k+1:n,k+1:n)-l(k,k+1:n)'*l(k,k+1:n);
        P(Pi(k),k)=1;
    end
    Am=l'*l;
    %exchange back the pivoted position
    G=P*Am*P';


    




