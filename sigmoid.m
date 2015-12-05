function [ out ] = sigmoid( in )
% compute sigmoid function
% Input:  in - input
% Output: out - output

out = 1 ./ (1 + exp(-in));

end

