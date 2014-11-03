function [ prob ] = prob_y_given_h( h, u , b_y )
%PROB_Y_GIVEN_H compute p(y|h).
%
%INPUT:
%   h = the values of hidden ,
%   u = the weights between the hidden and the class units,
%   b_y = the bias of the class units.
%OUTPUT:
%   prob = p(y|h).
%
%  prob_y_given_h( h, u , b_h ) returns the probability of the classification 
%  units y given the hidden units h, in a Classification RBM.

prob_not_norm = exp(bsxfun(@plus, u' * h, b_y));
norm_factor = sum(prob_not_norm);
prob = bsxfun(@rdivide, prob_not_norm, norm_factor);

end
