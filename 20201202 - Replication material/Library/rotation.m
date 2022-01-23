function rot=rotation(angle, p)
% ROTATION Returns a pxp rotation matrix constructed in the same way as in the GO-GARCH paper by van der Weide (2002).
%
% The function is ported from the Ox programming language, where it was
% implemented by Professor Heino Bohn Nielsen (UCPH)
%
% Simon Hetland, slh@econ.ku.dk, June 2018

%allocate storage for rotation matrix
rot = eye(p);

%Parameter counter
tal = 1;

for i=1:p-1
    for j=i+1:p
    %Construct pxp matrix with 2x2 rotation spanning one dimension
    help = eye(p);
    help(i,i) = cos(angle(tal));
    help(j,j) = cos(angle(tal));
    help(i,j) = sin(angle(tal));
    help(j,i) = -sin(angle(tal));
    
    %update rotation
    rot = rot*help;
    tal = tal +1;
    end
end


end