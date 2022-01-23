function EigenARCH_graphics(y, lambda, dates, single_graph)
%FUNCTION to graph rotated returns, y (pxT) and lambda (pxT)

set(0,'DefaultFigureWindowStyle','docked')

%Check inputs
switch nargin
    case 2
        dates =[];
    case 3
        single_graph=0;
    case 4    
        %do nothing
    otherwise
        error('Asset returns and array of covariance matrixes required.')
end


%Constants
T = length(y);
p = size(y,1);

if isempty(dates)
   dates = 1:T;    %If no dates, label x-axis by number 1:T
end

%%
if isequal(single_graph,0)
    figure('Name','Rotated returns','NumberTitle','on')
    %figure(n)
    for i=1:p
        subplot(p,1,i);plot(dates,y(i,:),'linewidth',0.5)
        title(['Asset # ', num2str(i)]) 
    end
    suptitle('Rotated returns, y_t');

    figure('Name','Eigenvalues','NumberTitle','on')
    %figure(n)
    for i=1:p
        subplot(p,1,i);plot(dates,lambda(i,:),'linewidth',0.5)
        title(['Asset # ', num2str(i)]) 
    end
    suptitle('Eigenvalues, \lambda_t');

    gamma = zeros(p,T);
    for i=1:T
        gamma(:,i) = lambda(:,i)./sum(lambda(:,i));
    end

    figure('Name','Eigenvalues % ','NumberTitle','on')
    %figure(n)
    for i=1:p
        subplot(p,1,i);plot(dates,gamma(i,:),'linewidth',0.5)
        title(['Eig # ', num2str(i)]) 
    end
    suptitle('Eigenvalues in percent');
else 
    figure('Name','Rotated returns','NumberTitle','on')
    %figure(n)
    plot(dates,y','linewidth',0.5);
    suptitle('Rotated returns, y_t');
    
    figure('Name','Eigenvalues','NumberTitle','on')
    plot(dates,lambda','linewidth',0.5)
    suptitle('Rotated returns, y_t');
    
    gamma = zeros(p,T);
    for i=1:T
        gamma(:,i) = lambda(:,i)./sum(lambda(:,i));
    end

    figure('Name','Eigenvalues % ','NumberTitle','on')
    plot(dates,gamma','linewidth',0.5)      
    suptitle('Eigenvalues in percent');
end

end