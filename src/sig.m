function [ sig ] = sig( x )
%SIG signomoid function
%
%INPUT:
%   x = the data vector.
%
%OUTPUT:
%   sig = the element by element signomoid of x.
%
%   sig( x ) returns 1 ./ (1 + exp(x))

sig = 1 ./ (1 + exp(-x));

end

