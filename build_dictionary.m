function [ dict_high, dict_low ] = build_dictionary(...
    patches_high, patches_low, dict_size )
% Build coupled dictionary from patches
% Input:  patches_high - list of high resolution patches
%         patches_low - list of low resolution patches
%         dict_size - size of coupled dictionary
% Output: dict_high - high resolution dictionary
%         dict_low - low resolution dictionary

[idx, dict_low] = kmeans(patches_low', dict_size);
dict_low = dict_low';

dict_high = zeros(size(patches_high, 1), dict_size);
for i = 1:dict_size
    dict_high(:, i) = mean(patches_high(:, idx==i), 2);
end
    
end