%% Housekeeping
clc
clear
format longG

addpath('../Data')
addpath('Library/')
addpath('Library/DERIVESTsuite/')

%%
%US Banks
Data     = readtable('USBanks.xlsx');        
Data = [Data(:,1),Data(:,2),Data(:,4),Data(:,5), Data(:,3)];
    
T_start = 1511;                 %3 jan 2006
T_end   = 4531;                 %2 jan 2018
    
    

%% Load data and specify options
type  = 'full';   %DONT CHANGE                                     %'scalar', 'diagonal', 'full', 'reduced rank'
est   = 'constrained';                                 %'constrained' or 'unconstrained'
GARCH = 1;                                             % 1 or 0 for GARCH or ARCH

p    = 3;                                              % # of variables
 
x     = table2array(Data(T_start:T_end,2:1+p))';     
dates = table2array(Data(T_start:T_end,1))';
T     = length(x);                                   

%Construct delta for k grid points in each delta_i, i=1,2,3 and make
%big matrix d, which we can iterate over.
%For k grid points    
k = 20;
gridnr =round([0:1/(k-1):1],4);
gridnr(end) = gridnr(end)-0.01;
d1 = gridnr; %B13    
d2 = gridnr; %B23
d3 = gridnr; %B33

d = zeros(3,k^3);
tal =1;
for i=1:k
    for j=1:k            
        for l=1:k
            d(:,tal) = [d1(i); d2(j); d3(l)];
            tal=tal+1;
        end
    end
end

%% UNRESTRICTED MODEL
tic

n=3;
npar_unres = 21;

L_vec_unres = zeros(k^3,1);
theta_vec_unres = zeros(k^3,npar_unres);
hess_res = zeros(npar_unres^2,k^3);
jac_res = zeros(npar_unres,T,k^3);

bad_init = zeros(k^3,1);
sing_hess = zeros(k^3,1);

parfor i=1:k^3
     theta0 = [0.290164957204009; 0.743467604566341; 0.798435252964075;         %phi
              0.0632133418206065; 0.00946867786463374; 6.49380786783991e-05;    %w
              0.122634736588721;0.0109656977153678;0.524026442390364;           % A col 1
              0.150602620999137;0.0247494920974897;0.0980399289109753;          % A col 2                 
              0.0103051496149450;0.00449351089681442;0.127657776100936;         % A col 3
              0.0246446631017461;6.20218659308590e-06;0.0217179236276176;       % B row 1
              0.818507945310757;0.884215605454200;11.7431367710272];            % B row 2
         
    
    %Ensure initial values work
    L_init=sqrt(-1);
    while imag(L_init)==1 || isnan(L_init)==1 || isinf(L_init)==1
        [L_init, ~, ~, ~] = loglikelihood(x, theta0, type,n,GARCH, d(:,i));
        hhelp=0;
        if isnan(L_init)==1 || imag(L_init)==1 || isinf(L_init)==1
            theta0 = 0.05*ones(npar_unres,1);
            bad_init(i)=1;
            hhelp=hhelp+1;
            if hhelp>1
                theta0 = 0.5*rand(npar_unres,1)
            end
        end
    end
    

    %Estimation
    %Minimize log likelihood function numerically
    options = optimset('Display','iter','PlotFcns',@optimplotfval,'UseParallel', true, 'TolFun',1e-6, 'TolX',1e-6, 'MaxFunEvals',100000, 'MaxIter', 5000);       

    %Constrained optimization
    lb = [ 0*ones(p*(p-1)/2,1); 0*ones(p,1)]; %Lower bound
    ub = [ pi/2*ones(p*(p-1)/2,1);  100*ones(p,1)]; %Upper bound
    if isequal(type,'full')
        lb = [lb; 0*ones(size(theta0,1)-size(lb,1),1)];
        ub = [ub; 100*ones(size(theta0,1)-size(ub,1),1)];
    end
  
    %FMINCON
    [theta, L_con, exit, output, L_con_mult, grad, hess] = fmincon(@(coef) -loglikelihood(x,coef(), type,n,GARCH, d(:,i)), theta0, [],[],[],[],lb,ub, [], options);
    L_vec_unres(i) = -T*L_con
    theta_vec_unres(i,:)=theta';
    
    %For kernel esetimation and simulation of crit values
 %   hess_unres(:,i) = hess(:)';    %Average hessian
 %   jac_unres(:,:,i)= jacobianest(@(coef) -loglikelihood_cont(x,coef(), type, n, GARCH,  d(:,i)),theta)'; %Array of jacobians for each t
    
    %sing_hess(i) = min(svd(hess))<10^(-8); %Check if hessian is singular
%    sing_hess(i) = rcond(reshape(hess_unres(:,i),21,21))<1e-12;
    %Takes the value 1 if hess is singular
    
end

%Find maximized LL value and associated parameter vector
[L_unres, d_nr_unres] = max(L_vec_unres)

%% RESTRICTED MODEL
n=2;
npar_res = 16;%size(theta0,1);

L_vec_res = zeros(k^3,1);
theta_vec_res = zeros(k^3,npar_res);
bad_init_res = zeros(k^3,1);
parfor i=1:k^3
%for i=1:k^3
    theta0 = [1.65547416002376e-06;0.904826053565072;0.579993724345270;1.60412003339580;0.0108954842054320;1.67332274923397e-05;0.00819130877472951;0.431345577291173;0.0323478557711055;0.212681211831140;0.00517725096716015;0.124661268337421;1.44032065052110e-06;1.04317772256559e-05;0.872621494119865;11.5384534948816];
    
    %Ensure initial values work
    L_init=sqrt(-1);
   while imag(L_init)==1 || isnan(L_init)==1 || isinf(L_init)==1
        [L_init, ~, ~, ~] = loglikelihood(x, theta0, type,n,GARCH, d(:,i));
        hhelp=0;
        if isnan(L_init)==1 || imag(L_init)==1 || isinf(L_init)==1
            theta0 = 0.05*ones(npar_res,1)
            bad_init_res(i) = 1;
            hhelp = hhelp+1;
            if hhelp>1
                theta0 = 0.5*rand(npar_res,1);
            end
        end
    end

    % Estimation
    %Minimize log likelihood function numerically
    options = optimset('Display','iter','PlotFcns',@optimplotfval,'UseParallel', true, 'TolFun',1e-8, 'TolX',1e-8, 'MaxFunEvals',100000, 'MaxIter', 5000);       

    %  Constrained optimization
    lb = [ 0*ones(p*(p-1)/2,1); 0*ones(p,1)]; %Lower bound
    ub = [ pi/2*ones(p*(p-1)/2,1);  100*ones(p,1)]; %Upper bound
    if isequal(type,'full')
        lb = [lb; 0*ones(size(theta0,1)-size(lb,1),1)];
        ub = [ub; 100*ones(size(theta0,1)-size(ub,1),1)];
    end
  
    %FMINCON
    [theta, L_con, exit, output, L_con_mult, grad, hess] = fmincon(@(coef) -loglikelihood(x,coef(), type,n,GARCH, d(:,i)), theta0, [],[],[],[],lb,ub, [], options);
    L_vec_res(i) = -T*L_con
    theta_vec_res(i,:)=theta';
    
    %For kernel esetimation and simulation of crit values
    theta_star = [theta(1:6); 0; theta(7:8); 0; theta(9:10); 0; theta(11:12); 0; theta(13:14); 0; theta(15:16)];
    hess = real(hessian(@(coef) -loglikelihood(x,coef(), type, 3, GARCH,  d(:,i)),theta_star)'); %Hessian
    hess_res(:,i) = hess(:)';
    jac_res(:,:,i)= real(jacobianest(@(coef) -loglikelihood_cont(x,coef(), type, 3, GARCH,  d(:,i)),theta_star)'); %Array of jacobians for each t
    
    %sing_hess(i) = rcond(reshape(hess_res(:,i),21,21))<1e-12;
    
end

%Find maximized LL value and associated parameter vector
[L_res, d_nr_res] = max(L_vec_res)

% supLR test:
supLR = 2*(L_unres-L_res) %Test value

%Copy whole matrices just in case
d_copy = d;
hess_res_copy = hess_res;
jac_res_copy =  jac_res;

%% Filenames 
k_str = num2str(k);
filename_hess = 'hess_res_row1_k';
filename_hess = [filename_hess, k_str, '.mat'];
 
filename_jac = 'jac_res_row1_k';
filename_jac = [filename_jac, k_str, '.mat'];

filename_d = 'd_res_row1_k';
filename_d = [filename_d, k_str, '.mat'];

filename_L_res = 'L_res_row1_k';
filename_L_res = [filename_L_res, k_str, '.mat'];

filename_L_unres = 'L_unres_row1_k';
filename_L_unres = [filename_L_unres, k_str, '.mat'];

%% Save jacobians, hessian and d 
save(filename_hess,'hess_res_copy')
save(filename_jac,'jac_res_copy')
save(filename_d, 'd_copy')
save(filename_L_res, 'L_vec_res')
save(filename_L_unres, 'L_vec_unres')

%% Load data
load(filename_hess)
load(filename_jac)
load(filename_d)
load(filename_L_res)
load(filename_L_unres)

npar_res = 16;
npar_unres = 21;

%%
for i=1:k^3
    sing_hess(i) = rcond(reshape(hess_res_copy(:,i),21,21))<1e-12;
end

%%
%Remove instances of singular hessian matrix or very low likelihood value
%(indicates numerical problems in estimation)
d = zeros(3,k^3-sum(sing_hess));
hess_res = zeros(npar_unres*npar_unres,k^3- sum(sing_hess));
jac_res = zeros(npar_unres, T,k^3- sum(sing_hess));

tal=1;
for i=1:k^3
   %if sing_hess(i) == 0 && L_vec_res(i)+100>=max(L_vec_res) 
   %if sing_hess(i) == 0 && L_vec_res(i)+10>=max(L_vec_res) 
   if sing_hess(i) == 0 && L_vec_res(i)+5>=max(L_vec_res) 
       d(:,tal) = d_copy(:,i);
       hess_res(:,tal) =  hess_res_copy(:,i);
       jac_res(:,:,tal) = jac_res_copy(:,:,i);
       tal=tal+1;
   end
end

%Update size of matrices
d = d(:,1:tal-1);
hess_res = hess_res(:, 1:tal-1);
jac_res = jac_res(:,:, 1:tal-1);

%Update dim of grid to contain invertible matrices
k3 = size(d,2);

t1 = toc/60

%% STEP 1 - Estimate the hessian and the outer product of the scores
%Hessian is given in hess_unres for the unrestricted model
tic
%K matrix - K\theta = \theta_1, where \theta_1 are parameters on the (potentially) boundary
K = zeros(5, npar_unres);
K(1, 7)  = 1; %A_11
K(2, 10) = 1; %A_12
K(3, 13) = 1; %A_13
K(4, 16) = 1; %B_11
K(5, 19) = 1; %B_12

%jac_unres is theta x T x k^3
%hess_unres is theta^2xk^3 (vectorized, use reshapce(hess_unres,size(theta0,1),size(theta0,1)) to get back

Sigma_Z = zeros(5*k3,5*k3);%Covariance matrix to sample from
tal_i =1; %Counter for loop over i
tal_j =1; %Counter for loop over j

%Fill out Sigma_Z row by row
for i =1:k3
   for j=1:k3
       hess_i = reshape(hess_res(:,i),npar_unres,npar_unres);
       hess_j = reshape(hess_res(:,j),npar_unres,npar_unres);

       Sigma_dd_tmp = 1/T*(jac_res(:,:,i)*jac_res(:,:,j)'); %Outer product of the scores for d_i d_j, vectorized
       Sigma_Z(tal_i:tal_i+4,tal_j:tal_j+4) = K*inv(hess_i)*Sigma_dd_tmp*inv(hess_j)*K'; %Sigma_Z_didj 
       %Sigma_Z(tal_i:tal_i+4,tal_j:tal_j+4) = K*pinv(hess_i)*Sigma_dd_tmp*pinv(hess_j)*K'; %Sigma_Z_didj 

       tal_j=tal_j+5; %Update counter
   end
   tal_i=tal_i+5; %Update counter
   tal_j=1;
end

% Reweight / slight numerical trouble making the matrix not exactly
% symmetric (very! small error)
Sigma_Z = 1/2*(Sigma_Z+Sigma_Z');
t2 = toc/60

%% STEP 2 - 5 Simulate and draw from limiting distribution
tic
M=10000;

asymp_dist = zeros(M,1);

%Construct Sigma_Z^{1/2}
%[Vec_Z, Val_Z] = eigs(Sigma_Z);
%Sigma_Z_half = Vec_Z*Val_Z.^(1/2)*Vec_Z';
%Sigma_Z_half = sqrtm(Sigma_Z);
[Vec1_Z, Val_Z, Vec2_Z] = svd(Sigma_Z);
Sigma_Z_half = Vec1_Z*Val_Z.^(1/2)*Vec2_Z';
   
parfor m=1:M
   rng(m+1352);
   %Step 2 - Draw from Z, k^3 times
   Z_d =  Sigma_Z_half*randn(5*k3,1);
   
   %Step 3 - Solve constrained optimization problem in eq (4)
   lambda_d = zeros(5,k3);
   tal=1;
   for i=1:k3       
       %Quadratic programming, minimize for eta 
       lambda_d_0=0.01*ones(5,1);
       lb_d = zeros(5,1);
       ub_d = 10000*ones(5,1);

       inv_hess = inv(reshape(hess_res(:,i), npar_unres, npar_unres))
       inv_big = inv(K*inv_hess*K');
       %inv_hess = pinv(reshape(hess_unres(:,i), npar_unres, npar_unres))
       %inv_big = pinv(K*inv_hess*K');
       
       [lambda_d(:,i)] = fmincon(@(coef)...
           (coef()-Z_d(tal:tal+4))'*inv_big*(coef()-Z_d(tal:tal+4))...
           , lambda_d_0, [],[],[],[],lb_d,ub_d, []);  
              
       tal=tal+5;
   end
   
   %Step 4 - Draw from the limiting distribution
   tmp1 = zeros(k3,1);
   for i =1:k3
       inv_hess1 = inv(reshape(hess_res(:,i),npar_unres, npar_unres));
       inv_big1 = inv(K*inv_hess1*K');
       %inv_hess1 = pinv(reshape(hess_unres(:,i),npar_unres, npar_unres));
       %inv_big1 = pinv(K*inv_hess1*K');
       tmp1(i) = lambda_d(:,i)'*inv_big1*lambda_d(:,i);
   end   
   asymp_dist(m) = max(tmp1);
end

t3 = toc/60
% Step 5 - compute critical value and print supLR test value
CV = quantile(asymp_dist,0.95)
supLR

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%              FUNCTIONS START HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [L, sigma2, lambda, persistence] = loglikelihood(x, param, type, n,  GARCH, param_fixed)
% Log likelihood function for the EigenARCH(1,1) moodel
% Inputs: 
%   x: pxT matrix of asset returns
%   param: vector of initial parameters
%   type: 'diagonal' or 'full'
%   n: the number of factors (n<=p)
%   var_est: estimated or targeting
%   GJR: 1 or 0 for asymmetries
%   GARCH: 0 if only ARCH model, 1 otherwise 
% Outputs:
%   L: Log likelihood value
%   sigma2: Array of filtered covariance matrices
%   lambda: pxT matrix of time varying eigenvalues
%   persistence: scalar indicating persistence of the process.

if nargin<5 || isequal(GARCH, [])
    GARCH = 1;
end
if nargin<6 || isequal(param_fixed, [])
    param_fixed = [];
end

[L_cont,sigma2, lambda, persistence] = loglikelihood_cont(x,param,type,n, GARCH,param_fixed);

T = length(x);
p = size(x,1);

L = 1/T*sum(L_cont);
end

function [L, sigma2, lambda, persistence] =loglikelihood_cont(x, param, type, n, GARCH,param_fixed)
% Log likelihood contributions for the EigenARCH(1,1) moodel
% Inputs: 
%   x: pxT matrix of asset returns
%   param: vector of initial parameters
%   type: 'diagonal' or 'full'
% Outputs:
%   L: Log likelihood value
%   sigma2: Array of filtered covariance matrices
%   lambda: pxT matrix of time varying eigenvalues
%   persistence: scalar indicating persistence of the process.

if nargin<5 || isequal(GARCH, [])
    GARCH = 1;
end

if nargin<6 || isequal(param_fixed, [])
    param_fixed = [];
end

if size(param,2)>size(param,1)
    param=param';
end

param = [param;param_fixed];

%Constants
T = length(x);
p = size(x,1);

%Parameters
tal=1; %Counter

V = rotation(param(tal:tal + p*(p-1)/2-1),p); %Rotation matrix
tal=tal+p*(p-1)/2;

omega = (param(tal:tal+p-1));   %Intercept in GARCH
tal=tal+p;
gamma = [V(:); omega];
    
if isequal(type,'full')
    alpha = zeros(p,p);
    if n<p
        alpha(2:end,1:p)     = reshape(param(tal:tal+n*p-1),n,p); %ARCH matrix
        tal = tal + n*p;    
    else
        alpha(1:n,1:p)     = reshape(param(tal:tal+n*p-1),n,p); %ARCH matrix
        tal = tal + n*p;    
    end
    
    beta = zeros(p,p);
    if GARCH ==1                
        if n<p
            beta(2:end,1:n) = reshape(param(tal:tal+n^2-1),n,n);      %GARCH matrix  
        else
            beta(1:n,1:n) = reshape(param(tal:tal+n^2-1),n,n);      %GARCH matrix  
        end
        
        tal = tal + n*n;
        gamma = [gamma; alpha(:); beta(:)];    
        
        %Grid paramters under H0
        if n<p
            beta(:,p) = param(tal:end);
        end
    end
else 
    error('doestnt work for other types than full')    
end


%Construct rotated returns
y               = V'*x; 
help =  eigs(cov(x'),p);

    
%Log likelihood
loglike           = zeros(1, T);         %Vector to hold log-likelihood contributions
sigma2            = zeros(p, p,  T);   %Array to contain time-varying covariance matrices
lambda            = zeros(p,T);        %\lambda (vector), contains time-varying eigenvalues
lambda(:,1)       = omega;


for i = 2:T 
    lambda(:,i) = omega+alpha*y(:,i-1).^2 + beta*lambda(:,i-1);               
    sigma2(:,:,i) = V*diag(lambda(:,i))*V'; %Save covariance estimate    
    sigma2(:,:,i) = 1/2*(sigma2(:,:,i)+sigma2(:,:,i)'); 
  
    loglike(i) = -p/2*log(2*pi)-0.5*sum(log(lambda(:,i)))-0.5*y(:,i)'*diag(1./lambda(:,i))*y(:,i);
    
end

persistence = max(eig(alpha+beta)); %persistence of stochastic process

L = loglike; %Returns vector of loglikelihood contributions

end
