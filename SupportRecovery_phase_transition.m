%%
%
% Problem: Samples x_1,...,x_n from R^d sharing a common support
% S=supp(x_i)\subset [d] such that |S|=k
% Given noisy linear measurements y_i = \Phi_i x_i + w_i, with \Phi \in \R^{mxd}
% such that m < k < d, recover the common support S.

% Input generation: A support S is drawn u.a.r. from all k-sized subsets of
% [d].
% x_i are generated i.i.d from
% (i) the mutlivariate Guassian N(0,diag(lam)), where lam \in R^d has
% support S, with nonzero entries drawn independently and uniformly from [lam_min,lam_max]
% (ii) the Rademacher distribution, and have nonzero entries restricted to S that take values from {-1,1} w.p. 1/2

% \Phi_i has entries drawn i.i.d. from N(0,1/m), and are independent across
% i
%
% Noise w_i is drawn i.i.d. from N(0,\sigma^2 I)

% For a given set of parameters (m,k,d,n,\sigma^2,lam_min,lam_max), this code runs iter iterations of the problem instance and records the
% success rate of the closed form estimator proposed in [1]. An instance is declared
% successful if the estimate S_hat = S.

% The success rate is plotted against the normalized number of samples,
% i.e. against n/n0, where n0 = (k^2*log(k*(d-k)))/m^2 is the scalinf
% sample complexity derived in [1].

% [1] Sample-measurement tradeoff in support recovery under a subgaussian prior
% (L. Ramesh, C. R. Murthy, and H. Tyagi. ISIT 2019)

%%

d=100;
m=2;
k=10;
sig2=0.1;
opt=1;                                      % opt=0: x_i are iid from N(0,diag(lam))
                                            % opt=1: x_i are iid with Ber(p)-1 entries

iter=100;                                    % iter: number of iterations for each problem instance

lam_min=1;
lam_max=2;

n0=(k^2*log(k*(d-k)))/m^2;

x_min=5;
x_max=40;
num_x=10;

scale=linspace(5,40,20);                    % points on the x axis, representing the ratio n/n0

n=ceil(n0.*scale);

success=zeros(1,length(scale));

for t=1:length(scale)
    
    for p=1:iter
        
        Phi=(1/sqrt(m)).*randn(m,d,n(t));
        y=zeros(m,n(t));
        
        [x,S]=generate_x(k,d,n(t),opt,lam_min,lam_max);
        
        w=(mvnrnd(zeros(1,m),sig2.*eye(m),n(t)))';
        
        for j=1:n(t)
            y(:,j)=Phi(:,:,j)*x(:,j)+w(:,j);
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        lam_est=CF_noisy(Phi,y);
        
        [mag,ind]=sort(lam_est,'descend');
        
        S_hat=ind(1:k);
        S_hat=reshape(S_hat,size(S));
        
        success(t)=success(t)+double(isequal(sort(S),sort(S_hat)));
        
        fprintf('Number of MMVs=%d, Monte-Carlo iteration number %d\n',n(t),p);
    end
    success(t)=success(t)/iter;
end

supp_vec_true=zeros(1,d);
supp_vec_true(S)=1;

supp_vec_est=zeros(1,d);
supp_vec_est(S_hat)=1;

figure(1)
stem(1:d,supp_vec_true,'go-')                                               % plot of true and estimated supports
hold on
stem(1:d,supp_vec_est,'r*-')

figure(2)
plot(scale,success,'r*-')                                                   % plot of success rate against nomalized n
title( ['d=' num2str(d) '    k=' num2str(k) '   m=' num2str(m)])
xlabel('$$\frac{n}{k^{2}\log k(d-k)/m^{2}}$$','Interpreter','latex','FontSize',12)
ylabel('Probability of exact support recovery','FontSize',12);

