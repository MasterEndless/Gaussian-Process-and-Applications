function [ T ] = Lanczos_Algorithm( alpha,beta )
    col=size(alpha,2);  %iteration
    row=size(alpha,1);  %third-dimension of T
    T=zeros(col,col,row);
    for i=1:row
        for j=1:col
            if j==1
                T(1,1,i)=1/alpha(i,1);
            else
                T(j,j,i)=1/alpha(i,j)+beta(i,j-1)/alpha(i,j-1);
                T(j-1,j,i)=sqrt(beta(i,j-1))/alpha(i,j-1);
                T(j,j-1,i)= T(j-1,j,i);
            end
        end
    end
end

