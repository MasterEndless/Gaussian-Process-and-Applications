B = linspace(-20,20,900);
A=rand(30);
A=A'*A;
C=0.01*eye(30);
D=A+C;
mvm_A=D+diag(0.01.*ones(1,30));
b2=linspace(-5,5,30);
b=reshape(b2,[30,1]);

P=inv(Pivoted_Cholesky_Composition(D));
I=eye(30);

pre=Standard_PCG(mvm_A,b,P);
non_pre=Standard_PCG(mvm_A,b,I);





