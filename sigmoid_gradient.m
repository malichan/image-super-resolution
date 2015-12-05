function [ out ] = sigmoid_gradient( in )
% compute gradient of sigmoid function
% Input:  in - input
% Output: out - output

out = sigmoid(in);
out = out .* (1 - out);

end

