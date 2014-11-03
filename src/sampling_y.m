function [ samp ] = sampling_y( prob )
%SAMPLING_Y sampling the softmax units y.
%
%INPUT:
%   prob = it's a matrix where each column is a random vector (the sum of it 
%   element must be 1) 
%OUTPUT:
%   samp = it's a matrix where each column is Gibbs samplig of the
%   corresponding column of prob.


batch_size = size(prob,2);
upb = cumsum(prob);
lowb = circshift(upb,1);
lowb(1,:) = zeros(1,batch_size);
r = rand(1,batch_size);
samp = bsxfun(@lt,r, upb) & bsxfun(@ge,r, lowb);

end

