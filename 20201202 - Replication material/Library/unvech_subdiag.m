function unvech_subdiag = unvech_subdiag(vecA,p)
% Function populate the subdiagonal of a pxp matrix with the elements in vecA
% vecA should be p*(p-1)/2.
%
% 
% Author: Simon Hetland, 
%[M,N] = size(vecA);

%if (M == N)
unvech_subdiag1  = diag(diag(ones(p,p)));
tal=1;
for j=2:p    
    unvech_subdiag1(j:p,j-1) = vecA(tal:tal+p-j);
    tal=tal+p-j+1;
end
    
    %for ii=2:M
    %    unvech_subdiag = [unvech_subdiag; matA(ii:end,ii-1)];
    %end
%else
     %error('Input must be a symmetric matrix.')
%end

unvech_subdiag = unvech_subdiag1;
end