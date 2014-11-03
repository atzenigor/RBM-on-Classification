function [ prob ] = prob_x_given_h( h, w, b_x )
%PROB_H_GIVEN_XY compute p(x|h).
%
%INPUT:
%   h = the values of hidden units,
%   w = the weights between the hidden and the input units,
%   b_x = the bias of the input units.
%OUTPUT:
%   prob = p(x|h)
%
%  prob_x_given_h( h, w, b_x ) returns the probability of the hidden 
%  units h given the class units y and the input units x, in a
%  Classification RBM.


prob = sig(bsxfun(@plus, w' * h, b_x));

end

