%Main program to calculate three quantities
function [ log_K_hat,B,U ] = Calculate_quantities( K,y,sigma,t )
B=Matrix_generator(K,y,t);
%M=inv(Pivoted_Cholesky_Composition(K,sigma));
n=size(K,1);
M=eye(n,n);
K_hat=K+sigma.^2*eye(n);
[U,T]=mPCG(K_hat,B,M);

%Calculate log|K_hat|
log_K_hat_sum=0;
for i=1:t
    log_K_hat_sum=log_K_hat_sum+B(:,i+1)'*logm(T(:,:,i))*B(:,i+1);
end
log_K_hat=log_K_hat_sum/t;







