function [ batch_data, batch_labels , batch_size ] = createBatches( data, labels, num_batches)
%CREATEBATCHES divide data and labels in batches.
%
%INPUT
%   data = dataset of the images,
%   labels = the vector of the labels,  
%   num_batches = the number of batches that have be created.
%OUTPUT
%   batch_data = the cell array with the batches of data,
%   batch_labels = the cell array with the batches of labels,
%   batch_size = the arity of the batches.
%
%  createBatches given the dataset data and the relative labels vector 
%  divide both in num_batches batches.The function produce num_batches-1
%  batches of batch-size size and one batch with the remainder elements
%  of data.

batch_size = ceil(size(data, 1) / num_batches);
batch_data = cell(1, num_batches);
batch_labels = cell(1, num_batches);

for i = 1:num_batches -1
   
    batch_data{i} = data((i-1) * batch_size + 1 : i*batch_size, : );
    batch_labels{i} = labels((i-1) * batch_size + 1 : i*batch_size, :);

end;

   batch_data{num_batches} = data((num_batches - 1)*batch_size + 1:end, : );
   batch_labels{num_batches} = labels( (i-1) * batch_size + 1 : i*batch_size,:);

end

