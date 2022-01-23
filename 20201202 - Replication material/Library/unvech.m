function Matrix = unvech(Vector)
% This function implements the unvech operator.
% INPUTS
%   Vector             [double]   a m*1 vector.
%
% OUTPUTS
%   Matrix             [double]   a n*n symetric matrix, where n solves n*(n+1)/2=m.
% Copyright (C) 2010 Dynare Team
 
 m = length(Vector);
 n = (sqrt(1+8*m)-1)/2;
 b = 0;
 Matrix = NaN(n,n);
 for col = 1:n
     idx = 1:col;
     Matrix(1:col,col) = Vector(b+idx);
     Matrix(col,1:col) = transpose(Matrix(1:col,col));
  b = b+length(idx);
 end