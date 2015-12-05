function [ weights_in, weights_out ] = initialize_neuralnet(...
    input_units, hidden_units, output_units, rand_range )
% initialize weight matrices of neuralnet
% Input:  input_units - number of input units
%         hidden_units - number of hidden units
%         output_units - number of output units
%         rand_range - range of random weights
% Output: weights_in - weights between input layer and hidden layer
%         weights_out - weights between hidden layer and output layer

weights_in = rand(input_units + 1, hidden_units) * rand_range * 2 - rand_range;
weights_out = rand(hidden_units + 1, output_units) * rand_range * 2 - rand_range;

end

