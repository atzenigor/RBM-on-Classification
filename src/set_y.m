function [ v_y ] = set_y( y, num_class )
%SET_Y return a matrix of labels v_y.
%
%INPUT:
%   y = a row vector of labels where each label is a number between 1 and
%      num_class,
%   num_class = a number of classes.
%OUTPUT:
%   v_y = the matrix in which each row represents one label expressed as a
%        vector where only the i-th element is 1 if it belongs to the i-th 
%        class, the other ones are 0.
%
%  set_y( y, num_class ) returns a matrix v_y in which every
%  row represents one labels in y.

s = size(y,1);
v_y = zeros(s, num_class);

for i = 1:s
    v_y(i,y(i)) = 1;
end

end

