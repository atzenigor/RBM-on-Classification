function W = randInitWeights(n_row, n_column)
%RANDINITIALIZEWEIGHTS Randomly initialize the elements of a matrix.
%
%INPUT:
%   n_row = the number of rows of the matrix that will be initialized,
%   n_column = the number of columns of the matrix that will be
%   initialized.
%OUTPUT:  
%   W = the random initialized matrix.
%
%  randInitWeights(n_row, n_column) initializes the 
%  matrix W with n_row rows and n_column columns, choosing the values 
%  in a random way from uniform samples in [-0,1; 0,1]. 

epsilon_init = 0.1;
W = rand(n_row, n_column) * 2 * epsilon_init - epsilon_init;

end
