function [ predictions ] = predict_neuralnet( instances, weights_in, weights_out )
% predict using neuralnet
% Input:  instances - instance matrix
%         weights_in - weights between input layer and hidden layer
%         weights_out - weights between hidden layer and output layer
% Output: predictions - prediction matrix

num_instances = size(instances, 2);

outs_input = [ones(1, num_instances); instances];
ins_hidden = weights_in' * outs_input;
outs_hidden = [ones(1, num_instances); sigmoid(ins_hidden)]; % sigmoid hidden units
ins_output = weights_out' * outs_hidden;
% outs_output = sigmoid(ins_output); % sigmoid output units
outs_output = identity(ins_output); % linear output units

predictions = outs_output;

end