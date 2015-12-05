function [ weights_in, weights_out ] = train_neuralnet(...
    weights_in, weights_out, instances, targets, learning_rate )
% train neuralnet
% Input:  weights_in - weights between input layer and hidden layer
%         weights_out - weights between hidden layer and output layer
%         instances - instance matrix
%         targets - target matrix
%         learning_rate - learning rate
% Output: weights_in - weights between input layer and hidden layer
%         weights_out - weights between hidden layer and output layer

num_instances = size(instances, 2);
input_units = size(weights_in, 1) - 1;
hidden_units = size(weights_in, 2);
output_units = size(weights_out, 2);

for i = 1:num_instances
    inst = instances(:, i);
    targ = targets(:, i);

    outs_input = [1; inst];
    ins_hidden = weights_in' * outs_input;
    outs_hidden = [1; sigmoid(ins_hidden)]; % sigmoid hidden units
    ins_output = weights_out' * outs_hidden;
    % outs_output = sigmoid(ins_output); % sigmoid output units
    outs_output = identity(ins_output); % linear output units

    errs_output = targ - outs_output;
    % errs_output = sigmoid_gradient(ins_output) .* errs_output; % sigmoid output units
    errs_output = identity_gradient(ins_output) .* errs_output; % linear output units
    errs_hidden = weights_out * errs_output;
    errs_hidden = sigmoid_gradient(ins_hidden) .* errs_hidden(2:end,:); % sigmoid hidden units

    weights_in = weights_in + learning_rate *...
        repmat(errs_hidden', input_units + 1, 1) .* repmat(outs_input, 1, hidden_units);
    weights_out = weights_out + learning_rate *...
        repmat(errs_output', hidden_units + 1, 1) .* repmat(outs_hidden, 1, output_units);
end

end