function [ predictions ] = predict_neuralnet( instances, weights_in, weights_out )

num_instances = size(instances, 2);

outs_input = [ones(1, num_instances); instances];
ins_hidden = weights_in' * outs_input;
outs_hidden = [ones(1, num_instances); sigmoid(ins_hidden)]; % sigmoid hidden units
ins_output = weights_out' * outs_hidden;
outs_hidden = sigmoid(ins_output); % sigmoid output units

predictions = outs_hidden;

end

