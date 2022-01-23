function vech_subdiag = vech_subdiag(matA)
% Function return elements from and below the main diagonal, then
% stacking by column to have a vector of size K*(K-1)/2.
%
% Author: Simon Hetland, 
[M,N] = size(matA);

if (M == N)
    vech_subdiag  = [];
    for ii=2:M
        vech_subdiag = [vech_subdiag; matA(ii:end,ii-1)];
    end
else
     error('Input must be a symmetric matrix.')
end
end