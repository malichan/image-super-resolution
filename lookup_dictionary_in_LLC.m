function [ patches_high ] = lookup_dictionary_in_LLC(...
    patches_low, dict_high, dict_low )
% Look up patches in coupled dictionary
% Input:  patches_low - list of low resolution patches
%         dict_high - high resolution dictionary
%         dict_low - low resolution dictionary
% Output: patches_high - list of high resolution patches

k_value = 5;

idx = knnsearch(dict_low', patches_low', 'K', k_value);

patch_number = size(patches_low, 2);
high_dimension = size(dict_high, 1);
patches_high = zeros(high_dimension, patch_number);

for i = 1 : patch_number
    patch = patches_low(:,i);
    current_idx = idx(i,:);
    current_subdict = dict_low(:,current_idx);
    % solve LLC
    % X w = b
    % subdict * w = patch
    w = current_subdict \ patch;
    
    patches_high(:,i) = dict_high(:, current_idx) * w;
end

end

