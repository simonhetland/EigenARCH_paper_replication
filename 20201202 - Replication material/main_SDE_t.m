%% Housekeeping
clc
clear
format longG

%addpath('../../Data')
addpath('Library/')
addpath('Library/DERIVESTsuite')

%US Banks
Data     = readtable('USBanks.xlsx');        
Data = [Data(:,1),Data(:,2),Data(:,4),Data(:,5), Data(:,3)];
    
T_start = 1511;                 %3 jan 2006
T_end   = 4531;                 %2 jan 2018        
    

%% Load data and specify options
p    = 3;                                             % # of variables
n    = 3;                                             % # of time-varying eigenvalues
 
x     = table2array(Data(T_start:T_end,2:1+p))';     
dates = table2array(Data(T_start:T_end,1))';
T     = length(x);       

%% 
% Summary statistics
fprintf('\nSummary statistics (average return and vol. in percent p.a.) \n')
fprintf('Correlations') 
corr_data = corr(x')

fprintf('Minimum and  maximum correlations')
[min(vech_subdiag(corr_data)), max(vech_subdiag(corr_data))]

fprintf('Average return and volatility (percent p.a.)') 
[mean(x')*252; std(x')*sqrt(252)]

% Principal components
[vec,val] = eigs(cov(x'),p)
val = diag(val);
val = 100.*val./sum(val);

fprintf('\nEigenvalues of unconditional covariance matrix are (percent)\n')
fprintf('%3.2f ',val)
fprintf('\n\nUnconditional variance explained by the first %d eigenvalue(s) %1.f percent\n',n, (100*sum(val(1:n))/sum(val)))
 
MGARCH_graphics(x,[],dates, 'desc'); % 'all', 'desc', 'res', 'vol'
 
%%
theta0 = 0.05*ones(p*(p-1)/2+p+2*n*p+(p-n)*n+1,1);
theta0(end)=5;
if n==2 && p==3
    theta0 = [-1.54560279172765;-0.0806774473215668;-0.0119487634011608;-4.38304390380264;-3.62008070610092;-2.66855195833823;0.190441238410221;0.424847445004956;-0.281445734509965;-0.0838089583717654;0.0387865708780291;0.257264159239584;1.00112751092523;0.173274548120554;0.404214938026934;-0.000343659979268894;0.848939848939107;-3.13062373940850e-05;0.000526790426836450;0.998217620563506;2.80038101484280];
elseif n==3 && p==3
    theta0 = [-1.468248031;	-0.113329795;	0.02070633;	-4.768142374;	-3.410457497;	-2.441715867;	0.156658817;	0.185293281;	0.414438456;	0.216908879;	-0.28746729;	0.000586877;	0.052157459;	0.050502011;	0.274535463;	0.98659817;	0.406293406;	0.000234004;	2.47E-05;	-0.841077895;	0.000187457;	0.041704293;	-9.79E-05;	0.997601005;	2.748972594];
end

%%
[L, ~, ~] = loglikelihood(x, theta0,n);  %Fetch log likelihood and the vector of variances 

 %% Estimation
%Minimize log likelihood function numerically
options = optimset('Display','iter','PlotFcns',@optimplotfval,'UseParallel', true, 'TolFun',1e-8, 'TolX',1e-10, 'MaxFunEvals',300000, 'MaxIter', 1000000);


lb = [0*ones(p*(p-1)/2,1); 0*ones(n,1)]; %Lower bound
ub = [ pi/2*ones(p*(p-1)/2,1);  100*ones(n,1)]; %Upper bound
lb = [lb; 0*ones(size(theta0,1)-size(lb,1),1)];
ub = [ub; 100*ones(size(theta0,1)-size(ub,1),1)];
lb(end) = 2;      %For v (degrees of freedom  
ub(end) = 1000;

[theta, L, exit, output, grad, hess] = fminunc(@(coef) -loglikelihood(x,coef(), n), theta0, options);  
%FMINCON
%[theta, L, exit, output, L_con_mult, grad, hess] = fmincon(@(coef) -loglikelihood(x,coef(), n), theta0, [],[],[],[],lb,ub, [], options);  

%%
%Fetch estimated covariances and persistence
[L, sigma2, lambda] = loglikelihood(x, theta,n);  %Fetch log likelihood and the vector of variances
L = T*L;
%%
%Fetch parameter estimates
[gamma, V, omega, alpha, beta,phi,g,a,b] = Eig_t_repar(p,n,theta);

% Optimality conditions
L_foc = output.firstorderopt   % Should be zero
L_soc = min(eig(hess))>0    % Should be 1 (only positive eigenvalues)


%% Model selections and misspecification testing
if 1==1
[aic_EigenARCH,bic_EigenARCH] = aicbic(L, size(theta,1), T) %Information criteria

% Compute residuals
res = zeros(p,T);
res2 = zeros(1,T);
for t = 2:T
    %compute residuals using ASYMMETRIC MATRIX SQUARE ROOT   
   [V_tmp,D]    = eig(sigma2(:,:,t));   
   res(:,t) = (diag(diag(D).^(-1/2))*V_tmp')*x(:,t);
   %res(:,t) = (diag(lambda(:,t).^(-1/2))*V')*x(:,t);
   
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
jac_t  = jacobianest(@(coef) -loglikelihood_cont(x,coef(), n), theta);                                                                
v_jac  = jac_t'*jac_t/T;         %divide by T to get 1/T sum dl/dp (dl/dp)'
se_jac = sqrt(diag(v_jac));

%Standard errors based on the sandwich formula
v_sandwich  = v_hess*v_jac*v_hess/T; %divide by T to get s.e. for \hat\theta~N(\theta_0,E/T)
se_sandwich = sqrt(diag(v_sandwich));
t_sandwich = theta./se_sandwich;

%Standard errors for transformed parameters based on the delta method
A = jacobianest(@(coef) Eig_t_repar_loc(p,n,coef()), theta);

v_delta = A*v_hess*A';
se_delta = sqrt(diag(v_delta)/T);
t_delta = gamma./se_delta;

v_delta_sandwich = A*v_sandwich*A';
se_delta_sandwich = sqrt(diag(v_delta_sandwich));

v_delta_jac = A*v_jac*A';
se_delta_jac = sqrt(diag(v_delta_jac));
t_delta_jac =gamma./se_delta_jac;

end


%% Print output nicely
disp('Estimation results')
[gamma, V, omega, alpha, beta, phi,g,a,b] = Eig_t_repar(p,n,theta);
phi
V
omega
g
a
b
alpha
beta
v=theta(end)+2

%% GRAPHS
EigenARCH_graphics(V'*x,lambda,dates,1);

MGARCH_graphics(x,sigma2(:,:,1:T),dates, 'vol'); % 'all', 'desc', 'res', 'vol'
MGARCH_graphics(x,sigma2(:,:,1:T),dates, 'res'); % 'all', 'desc', 'res', 'vol'

%% Residuals
quant = [0.01, 0.025, 0.05,0.25,0.5,0.75,0.95, 0.975, 0.99];
%quantiles = [quant;tinv(quant,v);norminv(quant);quantile(res',quant)']

quantiles = [quant;
             quantile((sqrt((v-2)/v)*trnd(v, 10^8,1)),quant);
             norminv(quant);
             quantile(res',quant)']


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%              FUNCTIONS START HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [L, sigma2, lambda] = loglikelihood(x, param, n)
% Log likelihood function for the EigenARCH(1,1) moodel
% Inputs: 
%   x: pxT matrix of asset returns
%   param: vector of initial parameters
%   n: the number of factors (n<=p)
% Outputs:
%   L: Log likelihood value
%   sigma2: Array of filtered covariance matrices
%   lambda: pxT matrix of time varying eigenvalues


[L_cont,sigma2, lambda] = loglikelihood_cont(x,param,n);

L = mean(L_cont);
end

function [L, sigma2, lambda] = loglikelihood_cont(x, param, n)
% Log likelihood contributions for the EigenARCH(1,1) moodel
% Inputs: 
%   x: pxT matrix of asset returns
%   param: vector of initial parameters
% Outputs:
%   L: Log likelihood value
%   sigma2: Array of filtered covariance matrices
%   lambda: pxT matrix of time varying eigenvalues

%Constants
T = length(x);
p = size(x,1);


[~, V, omega, alpha, beta] = Eig_t_repar(p,n,param);
    
%Rotated returns
y               = V'*x; 

loglike         = zeros(1, T);         %Vector to hold log-likelihood contributions
sigma2          = zeros(p, p,  T);   %Array to contain time-varying covariance matrices
lambda          = zeros(p,T);        %\lambda (vector), contains time-varying eigenvalues
lambda(:,1)     = omega;     %Initialize lambda in sample eigenvectors (sorted from largest to smallest)=
Lambda               = zeros(p, p,  T);   %\Lambda matrix, contains \lambda on the diagonal
Lambda(:,:,1) = diag(lambda(:,1));

%For t model
s = zeros(p,T);
S = zeros(p,p,T);
D = zeros(p,T);

%Degrees of freedom in t dist
%v = param(end);
v = param(end)+2;

if p==2
    G = [3,0,0,1;
         0,1,1,0;
         0,1,1,0;
         1,0,0,3];
     
     dLdl = [1,0;
             0,0;
             0,0;
             0,1];
elseif p==3
    G = [3,0,0,0,1,0,0,0,1;
         0,1,0,1,0,0,0,0,0;
         0,0,1,0,0,0,1,0,0;
         0,1,0,1,0,0,0,0,0;
         1,0,0,0,3,0,0,0,1;
         0,0,0,0,0,1,0,1,0;
         0,0,1,0,0,0,1,0,0;
         0,0,0,0,0,1,0,1,0;
         1,0,0,0,1,0,0,0,3];
     
    dLdl = [1,0,0;
            0,0,0;
            0,0,0;
            0,0,0;
            0,1,0;
            0,0,0;
            0,0,0;
            0,0,0;
            0,0,1];
end

vecI = eye(p);
vecI = vecI(:);
g = (v+p)/(v+2+p);


for i = 2:T 
    tmp = Lambda(:,:,i-1);
    
    %Weight term
    w = (v+p)/(v-2+y(:,i-1)'*diag(1./lambda(:,i-1))*y(:,i-1));    
    
    %Psi 
    Psi = kron(V,V)*dLdl;
    
    %Score
    D(:,i-1)   = 1/2*Psi'*...
               kron(V*diag(1./lambda(:,i-1))*V',V*diag(1./lambda(:,i-1))*V')*...
               (w.*kron(x(:,i-1),x(:,i-1))-kron(V,V)*tmp(:));    
    
    %Information
    I = 1/4*Psi'*kron(V*diag(lambda(:,i-1).^(-0.5))*V',V*diag(lambda(:,i-1).^(-0.5))*V')*...
        (g*G-vecI*vecI')*...
        kron(V*diag(lambda(:,i-1).^(-0.5))*V',V*diag(lambda(:,i-1).^(-0.5))*V')*Psi;
    
    %Scaling term
    S(:,:,i-1) = inv(I);
    
    %GAS term
    s(:,i-1) = S(:,:,i-1)*D(:,i-1);
    
    %Time-varying eig.
    lambda(:,i) = omega+alpha*s(:,i-1) + beta*lambda(:,i-1);
    
    %Eigenvalues and covariance matrix
    Lambda(:,:,i)     = diag(lambda(:,i));    
    sigma2(:,:,i)     = V*Lambda(:,:,i)*V';

    %log-likelihood cont.
    %loglike(i) = -p/2*log(2*pi)-0.5*sum(log(lambda(:,i)))-0.5*y(:,i)'*diag(1./lambda(:,i-1))*y(:,i);
    
    loglike(i) = gammaln((v+p)/2)-gammaln(v/2)-p/2*log((v-2)*pi)...
                 -1/2*log(det(Lambda(:,:,i)))-(v+p)/2*log(1+y(:,i)'*diag(1./lambda(:,i))*y(:,i)/(v-2));
end

L = loglike; %Returns vector of loglikelihood contributions

end

function [gamma, V, omega, alpha, beta, phi, g,a,b] = Eig_t_repar(p, n, param)
% Function to reparameterize the parameter-vector to the matrices

tal=1;
%Eigenvectors
phi = param(tal:tal + p*(p-1)/2-1);
phi = exp(phi)./(1+exp(phi)).*pi/2;
tal=tal+p*(p-1)/2;
V = rotation(phi,p); %Rotation matrix

%Constant
omega = exp(param(tal:tal+p-1));
%omega = (param(tal:tal+p-1));
tal=tal+p;

%Reduced rank matrices
a = vec2mat(param(tal:tal+p*n-1),n).^2;
%a = vec2mat(param(tal:tal+p*n-1),n);
tal = tal + p*n;
     
g = zeros(p,n);
g(1:p-n,:) = vec2mat(param(tal:tal+(p-n)*n-1),n).^2; %FIRST ROW FREE
%g(1:p-n,:) = vec2mat(param(tal:tal+(p-n)*n-1),n); %FIRST ROW FREE
g(p-n+1:end,:)=eye(n,n);
tal=tal+(p-n)*n;   

b = reshape(param(tal:tal+p*n-1),n,p)'.^2;
%b = reshape(param(tal:tal+p*n-1),n,p)';
tal=tal+p*n; 

alpha    = g*a'; 
beta    = g*b'; 

gamma = [V(:); omega(:); alpha(:); beta(:)];   

end
