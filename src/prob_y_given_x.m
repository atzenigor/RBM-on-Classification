function [ p_y_x ] = prob_y_given_x( w, u, b_y, b_h, x )
%PROB_Y_GIVEN_X return the p(y|x).
%
%INPUT:
%   w = weights between the hidden and the input units,
%   u = weights between the hidden and the classification units,
%   b_y = biases of the y units,
%   b_h = biases of the h units,
%   x = the input values.
%OUTPUT:
%   p_y_x = p(y|x).
%
%  prob_y_given_x( w, u, b_y, b_h, x ) returns the probability of the  
%  classification units y given the hidden units h, in a Classification RBM.

p_y_x = exp( b_y + sum (log (1 + exp(bsxfun(@plus, b_h + w * x, u))'), 2));
norm = sum(p_y_x) ;
p_y_x = p_y_x  ./ norm;

end

