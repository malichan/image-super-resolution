function [ patches_high ] = lookup_dictionary(...
    patches_low, dict_high, dict_low )
% Look up patches in coupled dictionary
% Input:  patches_low - list of low resolution patches
%         dict_high - high resolution dictionary
%         dict_low - low resolution dictionary
% Output: patches_high - list of high resolution patches

[pixel, patch_size] = size(patches_low);
patches_high = zeros(81, patch_size);

pre_part = pinv(dict_low' * dict_low) * dict_low';
size(pre_part)
for col = 1 : patch_size
    weight = pre_part * patches_low(:,col);
    patches_high(:,col) = dict_high * weight;
end


end

