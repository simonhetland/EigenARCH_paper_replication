%%%%%%%%%%%%%%%%%%%
%Recreates results for the unrestricted model

%% Housekeeping
clc
clear
format longG

addpath('Library/')
addpath('Library/DERIVESTsuite/')

%US Banks
Data     = readtable('USBanks.xlsx');        
Data = [Data(:,1),Data(:,2),Data(:,4),Data(:,5), Data(:,3)];
    
T_start = 1511;                 %3 jan 2006
T_end   = 4531;                 %2 jan 2018
    
    
%% Load data and specify options
p    = 3;       % # of variables
n    = 2;       % # of time-varying eigenvalues
 
x     = table2array(Data(T_start:T_end,2:1+p))';     
dates = table2array(Data(T_start:T_end,1))';
T     = length(x);                                   

%% 
% Summary statistics
% fprintf('\nSummary statistics (average return and vol. in percent p.a.) \n')
% fprintf('Correlations') 
% corr_data = corr(x')
% 
% fprintf('Minimum and  maximum correlations')
% [min(vech_subdiag(corr_data)), max(vech_subdiag(corr_data))]
% 
% fprintf('Average return and volatility (percent p.a.)') 
% [mean(x')*252; std(x')*sqrt(252)]
% 
% % Principal components
% [vec,val] = eigs(cov(x'),p)
% val = diag(val);
% val = 100.*val./sum(val);
% 
% fprintf('\nEigenvalues of unconditional covariance matrix are (percent)\n')
% fprintf('%3.2f ',val)
% fprintf('\n\nUnconditional variance explained by the first %d eigenvalue(s) %1.f percent\n',n, (100*sum(val(1:n))/sum(val)))
%  
%MGARCH_graphics(x,[],dates, 'desc'); % 'all', 'desc', 'res', 'vol'
 
%%
if n==p
    theta0 = 0.05*ones(21,1);            
    %theta0 = 0.05*ones(p*(p-1)/2+p+2*n*p+(p-n)*n,1);            
else
    theta0 = 0.05*ones(16,1);   
end
 
%% Check likelihood function
[L_initial, ~, ~, ~] = EigenARCH_loglikelihood(x, theta0,n);  %Fetch log likelihood and the vector of variances 

%% Estimation
%Minimize log likelihood function numerically
options = optimset('Display','iter','PlotFcns',@optimplotfval,'UseParallel', true, 'TolFun',1e-8, 'TolX',1e-10, 'MaxFunEvals',300000, 'MaxIter', 1000000);
lb = [0*ones(p*(p-1)/2,1); 0*ones(p,1); 0*ones(n*p,1); 0*ones(n*p,1);0*ones((p-n)*n,1)]; %Lower bound
ub = [ pi/2*ones(p*(p-1)/2,1);  100*ones(p,1); 100*ones(n*p,1); 100*ones(n*p,1);0*ones((p-n)*n,1)]; %Upper bound
[theta, L_con, exit, output, L_con_mult, grad, hess] = fmincon(@(coef) -EigenARCH_loglikelihood(x,coef(), n), theta0, [],[],[],[],lb,ub, [], options);
L_con = -T*L_con

%Fetch estimated covariances and persistence
[~, sigma2, lambda, persistence] = EigenARCH_loglikelihood(x, theta, n);  %Fetch log likelihood and the vector of variances
%%
%Fetch parameter estimates
[gamma, V, omega, alpha, beta] = EigenARCH_repar(p,n,theta);

% Optimality conditions
L_foc = output.firstorderopt   % Should be zero
L_soc = min(eig(hess))>0    % Should be 1 (only positive eigenvalues)

%% Model selections and misspecification testing
if 1==1
[aic_EigenARCH,bic_EigenARCH] = aicbic(L_con, size(theta,1), T) %Information criteria

% Compute residuals
res = zeros(p,T);
res2 = zeros(1,T);
for t = 2:T
    %compute residuals using ASYMMETRIC MATRIX SQUARE ROOT   
   [V_tmp,D]    = eig(sigma2(:,:,t));   
   res(:,t) = (diag(diag(D).^(-1/2))*V_tmp')*x(:,t);
   
   res2(:,t) = res(:,t)'*res(:,t);
end

% Misspecification tests
lag_test = [5,10,15]; %Which lags are we testing

no_aut      = zeros(p,size(lag_test,2));
pValue_aut  = zeros(p,size(lag_test,2));
stat_aut    = zeros(p,size(lag_test,2));
cValue_aut  = zeros(p,size(lag_test,2));

no_het      = zeros(p,size(lag_test,2));
pValue_het  = zeros(p,size(lag_test,2));
stat_het    = zeros(p,size(lag_test,2));
cValue_het  = zeros(p,size(lag_test,2));

no_arch      = zeros(p,size(lag_test,2));
pValue_arch  = zeros(p,size(lag_test,2));
stat_arch    = zeros(p,size(lag_test,2));
cValue_arch  = zeros(p,size(lag_test,2));

for i=1:p
    [no_aut(i,:),  pValue_aut(i,:),  stat_aut(i,:),  cValue_aut(i,:)] = lbqtest(res(i,:),    'lags', [5,10,15]);   %Ljung-Box test for residual autocorrelation (H0 is no autocorrelation)
    [no_het(i,:),  pValue_het(i,:),  stat_het(i,:),  cValue_het(i,:)] = lbqtest(res(i,:).^2, 'lags', [5,10,15]);   %Ljung-Box test for heteroscedasticity (H0 is no heteroscedasticity)
    [no_arch(i,:), pValue_arch(i,:), stat_arch(i,:), cValue_arch(i,:)] = archtest(res(i,:),    'lags', [5,10,15]); %Engle's no-arch test (H0 is no arch)
end
pValue_aut
pValue_het
pValue_arch
end

%% Inference
if 1==1
%Standard errors based on Hessian
v_hess = inv(hess);                 
se_hess = sqrt(diag(v_hess)/T);  % divide by T to get s.e. for \hat\theta~N(\theta_0,E/T)

%Standard errors based on outer product of scores
jac_t  = jacobianest(@(coef) -EigenARCH_loglikelihood_cont(x,coef(), n), theta);                                                                
v_jac  = jac_t'*jac_t/T;         %divide by T to get 1/T sum dl/dp (dl/dp)'

%Standard errors based on the sandwich formula
v_sandwich  = v_hess*v_jac*v_hess; %divide by T to get s.e. for \hat\theta~N(\theta_0,E/T)
se_sandwich = sqrt(diag(v_sandwich)/T);

%Standard errors for transformed parameters based on the delta method
A = jacobianest(@(coef) EigenARCH_repar(p,n,coef()), theta);

v_delta = A*v_hess*A';
se_delta = sqrt(diag(v_delta)/T);

v_delta_sandwich = A*v_sandwich*A';
se_delta_sandwich = sqrt(diag(v_delta_sandwich)/T);

end


%% Print output
disp('Estimation results')
[gamma, V, omega, alpha, beta, phi] = EigenARCH_repar(p,n,theta);
phi
V
omega
alpha
beta
persistence
lambda_var = diag((eye(p)-alpha-beta)^(-1)*omega)
uncon_var = V*diag((eye(p)-alpha-beta)^(-1)*omega)*V'

%% GRAPHS
EigenARCH_graphics(V'*x,lambda,dates,1);

MGARCH_graphics(x,sigma2(:,:,1:T),dates, 'vol'); % 'all', 'desc', 'res', 'vol'
MGARCH_graphics(x,sigma2(:,:,1:T),dates, 'res'); % 'all', 'desc', 'res', 'vol'

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
if n<p
    alpha = zeros(p,p);
    alpha(2:end,1:p)     = reshape(param(tal:tal+n*p-1),n,p); %ARCH matrix
    tal = tal + n*p;    
    beta = zeros(p,p);
    beta(2:end,1:2) = reshape(param(tal:end),n,2);      %GARCH matrix       
else
    alpha = zeros(p,p);
    alpha(1:end,1:p)     = reshape(param(tal:tal+n*p-1),n,p); %ARCH matrix
    tal = tal + n*p;    
    beta = zeros(p,p);
    beta(1:end,1:2) = reshape(param(tal:end),n,2);      %GARCH matrix  
end

gamma = [V(:); omega(:); alpha(:); beta(:)];   

end


