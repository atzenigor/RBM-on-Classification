function [ prob ] = prob_h_given_xy( x, y, w, u, b_h )
%PROB_H_GIVEN_XY compute p(h=1|x,y).
%
%INPUT
%   x = the values of input units,
%   y = the values of class units,
%   w = the weights between the hidden and the input units,
%   u = the weights between the hidden and the class units,
%   b_h = the bias of the hidden units.
%OUTPUT
%   prob = the probability p(h=1|x,y).
%
%  prob_h_given_xy( x, y, w, u, b_h ) returns the probability of the hidden 
%  units h given the class units y and the input units x, in a
%  Classification RBM.


prob = sigmf(bsxfun(@plus, w * x + u * y, b_h), [1 0]);

end

