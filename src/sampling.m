function [ samp ] = sampling( p )
%SAPLING do the sampling of p.
%
%INPUT:
%   p = the not binary values that have to be sampled.
%OUTPUT:
%   samp = the sampled values.
%
%  sampling( p ) returns a vector of binary samples, p can be a matrix.

samp = p > rand(size(p));

end

