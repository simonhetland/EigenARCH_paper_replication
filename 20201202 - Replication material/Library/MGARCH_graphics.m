function MGARCH_graphics(x, sigma2, dates, graph_type)
% MGARCH_GRAPHICS Plots standardized residuals, correlations and standard deviations for mulitivariate GARCH models. 

% Inputs are MGARCH_GRAPHICS(x, sigma2, dates, graph_type). 
% where 'x' is a p x T matrix of asset returns, 'sigma2' is a p x p x T array of covariance matrices, and 'dates' are an 
% optional input of 1 x T vector of dates, 'dates'. Finally, graph_type
% determines whether asset returns ('desc'), residuals ('res') and associated misspecification tests are
% printed, or if we print correlations and volatilities ('vol'). Make all
% graphs by entering 'all'

set(0,'DefaultFigureWindowStyle','docked')

%Check inputs
switch nargin
    case 2
        dates =[];
    case 3
        graph_type = 'all';
    case 4
        %do nothing
    otherwise
        error('Asset returns and array of covariance matrixes required.')
end



%Constants
T = length(x);
p = size(x,1);

if isempty(dates)
   dates = 1:T;    %If no dates, label x-axis by number 1:T
end

%% Plot asset returns and ACF
n=2;

if isequal(graph_type,'desc') || isequal(graph_type,'all')
    figure('Name','Log returns','NumberTitle','on')
    %figure(n)
    for i=1:p
        subplot(p,1,i);plot(dates,x(i,:),'linewidth',0.5)
        title(['Asset # ', num2str(i)]) 
    end
    %suptitle('Log-returns');
    n=n+1;


    figure('Name','ACF - log returns','NumberTitle','on')
    %figure(n)
    for i=1:p
        subplot(p,1,i);autocorr(x(i,:),40)
        title(['Asset # ', num2str(i)]) 
    end
    %suptitle('Autocorrelation of log returns');
    n=n+1;

    figure('Name','ACF - abs log returns','NumberTitle','on')
    %figure(n)
    for i=1:p
        subplot(p,1,i);autocorr(sqrt(x(i,:).^2),40)
        title(['Asset # ', num2str(i)]) 
    end
    %suptitle('Autocorrelation of squared log returns');
    n=n+1;
    
    % Plot histogram of returns and normal density
    figure('Name','Histogram - returns','NumberTitle','on')
    %figure(n)
    for i=1:p
        subplot(p,1,i); histfit(x(i,:))
        %histogram(res(i,:)); hold on;
    %    plot([-6:0.1:6],pdf(makedist('Normal'),[-3:0.1:3]),'LineWidth',2)
        title(['Asset # ', num2str(i)]) 
    end
    n=n+1;
    
end

%% Compute residuals, standard dev, and correlations
if isequal(graph_type, 'all') || isequal(graph_type, 'res') || isequal(graph_type, 'vol')
    res = zeros(p,T);
    stdev = zeros(p,T);
    correl = zeros(p*(p-1)/2,T);

    for t = 2:T
        %compute residuals using SYMMETRIC MATRIX SQUARE ROOT
       %[V,D]    = eigs(sigma2(:,:,t),p); 
       [V,D]    = eig(sigma2(:,:,t));   
       %res(:,t) = (V*diag(diag(D).^(-1/2))*V')*x(:,t);
       res(:,t) = (diag(diag(D).^(-1/2))*V')*x(:,t);

       %V*diag(diag(D).^(-1/2))*V'
       %x(:,t)

       %compute annualized standard deviation
       stdev(:,t)      = diag(sigma2(:,:,t)).^(1/2)*sqrt(252);

       %compute correlation matrix
       help   = diag(diag(sigma2(:,:,t)).^(-1/2))*sigma2(:,:,t)*diag(diag(sigma2(:,:,t)).^(-1/2)); 

       %only store correlations
       correl(:,t) = vech_subdiag(help);   

    end
end

%% %Plot estimation results
if isequal(graph_type, 'all') || isequal(graph_type, 'res') 
    % Plot residuals
    figure('Name','Std. residuals','NumberTitle','on')
    %figure(n)
    for i=1:p
        subplot(p,1,i);plot(dates,res(i,:),'linewidth',0.5)
        title(['Asset # ', num2str(i)]) 
    end
    %suptitle('Standarized residuals');
    n=n+1;

%     % Plot autocorrelation function of residuals
%     figure('Name','ACF - std. residuals','NumberTitle','on')
%     %figure(n)
%     for i=1:p
%         subplot(p,1,i);autocorr(res(i,:),40)
%         title(['Asset # ', num2str(i)]) 
%     end
%     %suptitle('Autocorrelation of std. residuals');
%     n=n+1;
    
    % Plot autocorrelation function of absolute residuals
    figure('Name','ACF - abs std. residuals','NumberTitle','on')
    %figure(n)
    for i=1:p
        subplot(p,1,i);autocorr(abs(res(i,:)),40)
        title(['Asset # ', num2str(i)]) 
    end
    %suptitle('Autocorrelation of std. residuals');
    n=n+1;    
    

    % Plot QQ plot of std. residuals
    figure('Name','QQ plot - std. residuals','NumberTitle','on')
    %figure(n)
    for i=1:p
        subplot(p,1,i);qqplot(res(i,:))
        title(['Asset # ', num2str(i)]) 
    end
    %suptitle('QQ plot of std. residuals');
    n=n+1;

    % Plot histogram of returns and normal density
    figure('Name','Histogram - std. residuals','NumberTitle','on')
    %figure(n)
    for i=1:p
        subplot(p,1,i); histfit(res(i,:))
        %histogram(res(i,:)); hold on;
    %    plot([-6:0.1:6],pdf(makedist('Normal'),[-3:0.1:3]),'LineWidth',2)
        title(['Asset # ', num2str(i)]) 
    end
    %suptitle('Density of std. residuals');
    n=n+1;
end

if isequal(graph_type, 'all') || isequal(graph_type, 'vol')
    % Plot standard deviations
    figure('Name','St. dev.','NumberTitle','on')
    %figure(n)
    for i=1:p
        subplot(p,1,i);plot(dates,stdev(i,:),'linewidth',0.5)
        title(['Asset # ', num2str(i)]) 
    end
    %suptitle('Annualized standard deviations');
    n=n+1;

    % Plot correlation matrix
    figure('Name','Correlations','NumberTitle','on')
    %figure(n)
    for i=1:p*(p-1)/2
        subplot(p*(p-1)/2,1,i);plot(dates,correl(i,:),'linewidth',0.5)
        title(['Correlation # ', num2str(i)]) 
    end
    %suptitle('Correlations')

end




end