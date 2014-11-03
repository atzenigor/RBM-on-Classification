function [ err ] = predict( testdata, testlabels, num_class, b_y, b_h, w, u )
%PREDICT predict on the test data returning the error
%
%INPUT:
%   testdata = the dataset with the images that will be classified,
%   testlabels = the labels of the images in testdata,
%   num_class = the number of different class,
%   b_y = bias of the hidden units,
%   b_h = bias of the classification units,
%   w = weight between hidden and input units,
%   u = weight between hidden and classification units.
%OUTPUT:
%   err = the percentage of wrong classifications.
%
%  predict( testdata, testlabels, num_class, b_y, b_h, w, u ) returns the 
%  error using the parameter of a RBM (b_y, b_h, w, u) compute the free energy
%  for each example in the test data, chose the y with the lower 
%  energy, compare each prediction with the test label, and find how many errors
%  occurred. 
  
  
  numcases= size(testdata, 1);
  w_times_x = w * testdata';
  neg_free_energ = zeros(num_class, numcases);
  for iClass= 1:num_class
    neg_free_energ(iClass,:) = b_y(iClass) + sum(log(1 + ...
      exp(bsxfun(@plus, b_h + u(:,iClass), w_times_x ))));
  end;
  [~, prediction] = max(neg_free_energ);
  err = sum(prediction ~= testlabels') / numcases * 100;

end