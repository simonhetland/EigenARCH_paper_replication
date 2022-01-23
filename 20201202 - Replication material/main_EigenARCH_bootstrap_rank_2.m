%% Housekeeping
clc
clear
format longG

addpath('Library/')
addpath('Library/DERIVESTsuite/')

%% Monte carlo settings
B=399;             %monte carlo iterations
LR = zeros(B,1);    %variable to hold LR test stats

bb = 1500;          %Estimations
temp = zeros(bb,1); 
FOC = zeros(bb,2); %Col 1 contains FOC for unrestricted estimation, col 2 for restricted
SOC = zeros(bb,2); %Col 1 contains SOC for unrestricted estimation, col 2 for restricted

%Restricted residuals
Data = readtable('residuals_rr2.xlsx');
res   = table2array(Data(1:end,2:4))';  
res   =  res-mean(res')';           %Center to mean zero
res   =  inv(chol(cov(res')))*res;  %Standardize to be D(0,I_3)

%Parameters
theta0   =  table2array(readtable('theta_rr2.xlsx'));  
p=3;
n0=2;
[~, V0, omega0, alpha0, beta0, phi0, g0, a0, b0] = EigenARCH_repar(p,n0,theta0);

theta_unres0 = [phi0(:);(omega0(:));(alpha0(:));(beta0(:))];
lb_unres = [0*ones(p*(p-1)/2,1);     0*ones(p,1)]; %Lower bound
ub_unres = [ pi/2*ones(p*(p-1)/2,1);  100*ones(p,1)]; %Upper bound
lb_unres = [lb_unres; 0*ones(size(theta_unres0,1)-size(lb_unres,1),1)];
ub_unres = [ub_unres;  100*ones(size(theta_unres0,1)-size(ub_unres,1),1)];
 
theta_res0 = theta0;
lb_res = [0*ones(p*(p-1)/2,1);     0*ones(n0,1)]; %Lower bound
ub_res = [ pi/2*ones(p*(p-1)/2,1);  100*ones(n0,1)]; %Upper bound
lb_res = [lb_res; 0*ones(size(theta_res0,1)-size(lb_res,1),1)];
ub_res = [ub_res;  100*ones(size(theta_res0,1)-size(ub_res,1),1)];
    
T = 3020; %Same size as original sample

%%
parfor j=1:bb    
%for j=1:B    
%j=1
    %% Simulate the data
    rng(1+j);   
    p=3;
    x       = zeros(p, T);
    y0 = zeros(p, T);
    lambda0  = zeros(p, T);
    Lambda0  = zeros(p, p, T);   
    sigma20  = zeros(p, p, T);
    loglike0 = zeros(1,T);    

    lambda0(:,1) = omega0;

    y0(:,1) = V0'*x(:,1);

    for i=2:T
        %Eigenvalues
        lambda0(:,i)   = omega0+alpha0*(y0(:,i-1).^2)+beta0*lambda0(:,i-1);
        Lambda0(:,:,i)     = diag(lambda0(:,i));

        %Covariance matrix
        L_inv         = diag(1./lambda0(:,i));
        sigma20(:,:,i) = V0*diag(lambda0(:,i))*V0';
        sigma20(:,:,i) = 1/2*(sigma20(:,:,i)+sigma20(:,:,i)');     

        %Simulate return
        [Vec, Val] = eig(sigma20(:,:,i));
        sqrtSigma = Vec*(Val.^(0.5))*Vec';
        x(:,i)   = sqrtSigma*res(:,randi(T));     
        %x(:,i)   = sqrtSigma*randn(p,1);
        y0(:,i) = V0'*x(:,i);

        loglike0(i) = -0.5*sum(log(lambda0(:,i)))-0.5*y0(:,i)'*L_inv*y0(:,i);    
    end

    L0 = sum(loglike0)-T*p/2*log(2*pi);


    %% ESTIMATE UNRESTRICTED MODEL
    % Specify options    
    p    = 3;                                             % # of variables
    n    = 3;                                             % # of time-varying eigenvalues

    [L_unres, ~, ~, ~] = EigenARCH_loglikelihood(x, theta_unres0,n);  %Fetch log likelihood and the vector of variances  

    %  Constrained optimization
   options = optimset('Display','iter','PlotFcns',@optimplotfval,'UseParallel', true, 'TolFun',1e-5, 'MaxFunEvals',300000, 'MaxIter', 1000000);    
   [theta_unres, L_unres, exit, output, L_con_mult, grad_unres, hess_unres] = fmincon(@(coef) -EigenARCH_loglikelihood(x,coef(), n), theta_unres0, [],[],[],[],lb_unres,ub_unres, [], options);
   L_unres = -T*L_unres;
  
    %Fetch parameter estimates
    [gamma, V, omega, alpha, beta,phi] = EigenARCH_repar(p,n,theta_unres);

    %% ESTIMATE RESTRICTED MODEL
    % Specify options
    p    = 3;                                             % # of variables
    n    = 2;                                             % # of time-varying eigenvalues

    [L_res, ~, ~, ~] = EigenARCH_loglikelihood(x, theta_res0,n);  %Fetch log likelihood and the vector of variances 
    %  Constrained optimization
    [theta_res, L_res, exit, output, L_con_mult, grad_res, hess_res] = fmincon(@(coef) -EigenARCH_loglikelihood(x,coef(),n), theta_res0, [],[],[],[],lb_res,ub_res, [], options);
    L_res = -T*L_res;
           
   FOC(j,:) = [max(grad_unres), max(grad_res)];
   SOC(j,:) = [min(eig(hess_unres))>0, min(eig(hess_res))>0];    % Should be 1 (only positive eigenvalues)   

    %% LR test
    temp(j) = 2*(L_unres-L_res);
    disp('iteration')
    disp(j)        
    
end
%%
tal=1;
i=1;
%Due to numerical problems, some LR tests are negative,
%keep first 399 LR tests that are strictly positive (i.e. w/o numerical
%problems)
while tal<=B && i<bb+1
   if temp(i)>0 && temp(i)<1000
       LR(tal) = temp(i);
       tal=tal+1;      
   end
   i=1+i;       
end


%%
%Critical value
Crit_LR_95 = quantile(LR,0.95)

%Compute empirical CDF (interpolated)
[f_LR, x_LR]   = ecdf(LR);

figure(1);
plot(x_LR,[f_LR, chi2cdf(x_LR,4)]);
title('chi2(4) CDF against LR CDF')
legend('LR CDF', 'chi2(4) CDF')

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%              FUNCTIONS START HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [L, sigma2, lambda, persistence] = EigenARCH_loglikelihood(x, param, n)
% Log likelihood function for the EigenARCH(1,1) moodel
% Inputs: 
%   x: pxT matrix of asset returns
%   param: vector of initial parameters
%   n: the number of factors (n<=p)
% Outputs:
%   L: Log likelihood value
%   sigma2: Array of filtered covariance matrices
%   lambda: pxT matrix of time varying eigenvalues
%   persistence: scalar indicating persistence of the process.

[L_cont,sigma2, lambda, persistence] = EigenARCH_loglikelihood_cont(x,param,n);

L = mean(L_cont);
end

function [L, sigma2, lambda, persistence] = EigenARCH_loglikelihood_cont(x, param, n)
% Log likelihood contributions for the EigenARCH(1,1) moodel
% Inputs: 
%   x: pxT matrix of asset returns
%   param: vector of initial parameters
%   n: the number of factors (n<=p)   
% Outputs:
%   L: Log likelihood value
%   sigma2: Array of filtered covariance matrices
%   lambda: pxT matrix of time varying eigenvalues
%   persistence: scalar indicating persistence of the process.

%Constants
T = length(x);
p = size(x,1);

[gamma, V, omega, alpha, beta] = EigenARCH_repar(p, n, param); %Fecth (reparameterized) parameter matrices

%Rotated returns
y               = V'*x; 
    
%Log likelihood
loglike         = zeros(1, T);         %Vector to hold log-likelihood contributions
sigma2          = zeros(p, p,  T);   %Array to contain time-varying covariance matrices
lambda          = zeros(p,T);        %\lambda (vector), contains time-varying eigenvalues
lambda(:,1)     = omega;


for i = 2:T 
    %conditional eigenvalues
    lambda(:,i) = omega+alpha*y(:,i-1).^2 + beta*lambda(:,i-1);               
    
    %conditional covariance matrix
    sigma2(:,:,i) = V*diag(lambda(:,i))*V'; %Save covariance estimate    
    
    %log-likelihood contributions
    loglike(i) = -p/2*log(2*pi)-0.5*sum(log(lambda(:,i)))-0.5*y(:,i)'*diag(1./lambda(:,i))*y(:,i);
    
end

persistence = max(eig(alpha+beta)); %persistence of stochastic process

L = loglike; %Returns vector of loglikelihood contributions

end

function [gamma, V, omega, alpha, beta, phi, g,a,b] = EigenARCH_repar(p, n, param)
% Function to reparameterize the parameter-vector to the matrices

tal=1;
%Eigenvectors
phi = param(tal:tal + p*(p-1)/2-1);
tal=tal+p*(p-1)/2;
V = rotation(phi,p); %Rotation matrix

%Constant
omega = (param(tal:tal+p-1));
tal=tal+p;

%Reduced rank matrices
a = vec2mat(param(tal:tal+p*n-1),n);
tal = tal + p*n;
     
g = zeros(p,n);
g(1:p-n,:) = vec2mat(param(tal:tal+(p-n)*n-1),n); %FIRST ROW FREE
g(p-n+1:end,:)=eye(n,n);
tal=tal+(p-n)*n;   

b = reshape(param(tal:tal+p*n-1),n,p)';
tal=tal+p*n; 

alpha    = g*a'; 
beta    = g*b'; 

gamma = [V(:); omega(:); alpha(:); beta(:)];   

end

