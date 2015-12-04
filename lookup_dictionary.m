function [ patches_high ] = lookup_dictionary(...
    patches_low, dict_high, dict_low )
% Look up patches in coupled dictionary
% Input:  patches_low - list of low resolution patches
%         dict_high - high resolution dictionary
%         dict_low - low resolution dictionary
% Output: patches_high - list of high resolution patches

idx = knnsearch(dict_low', patches_low');
patches_high = dict_high(:,idx);

end

