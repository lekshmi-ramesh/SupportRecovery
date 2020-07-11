function[x,S]=generate_x(k,d,n,opt,lam_min,lam_max)

p=.5;
x=zeros(d,n);                                   
S=randsample(d,k);                              

switch(opt)
    
    case 0
        lam=zeros(1,d);
        lam(S)=random('uniform',lam_min,lam_max,[1,k]);
        x(S,:)=(mvnrnd(zeros(1,k),diag(lam(S)),n))';
        
    case 1
        for i=1:n
        x(S,i)=(rand(k,1)>p);
        x(S,i)=2.*x(S,i)-1;
        end
end 
end