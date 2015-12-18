function [ patches_high] = lookup_dictionary_in_sparse_coding(...
    patches_low, dict_high, dict_low )
% Look up patches in coupled dictionary
% Input:  patches_low - list of low resolution patches
%         dict_high - high resolution dictionary
%         dict_low - low resolution dictionary
% Output: patches_high - list of high resolution patches

lam = 0.01;
patch_number = size(patches_low, 2);
high_dimension = size(dict_high, 1);
patches_high = zeros(high_dimension, patch_number);

for i = 1 : patch_number
    patch = patches_low(:, i);
    w = lasso (dict_low , patch, 'lambda', lam);
    patches_high(:,i) = dict_high * w;
end

end

