function[lam_est]=SupportRecovery(phi,y)
[m,d,n]=size(phi);

A=zeros(d,d);
z=zeros(d,n);
for i=1:n
  z(:,i)=phi(:,:,i)'*y(:,i);
  A=A+(z(:,i)*z(:,i)');
end
A=A./n;

lam_est=diag(A);

end