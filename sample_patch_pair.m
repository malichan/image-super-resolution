function [ patches_high, patches_low ] = sample_patch_pair(...
    images_high, images_low, patch_size, scale_factor, sample_size )
% Sample patches from images
% Input:  images - cell of image matrices
%         patch_size - patch size
%         sample_size - sample size
% Output: patches - list of patches

num_images = size(images_high, 2);
patch_size_hi = patch_size * scale_factor;

patches_high = zeros(patch_size_hi * patch_size_hi, sample_size);
patches_low = zeros(patch_size * patch_size, sample_size);
for k = 1:sample_size
    image_idx = ceil(rand() * num_images);
    
    [height, width] = size(images_low{image_idx});
    i = ceil(rand() * (height - patch_size + 1));
    j = ceil(rand() * (width - patch_size + 1));
    patch_low = images_low{image_idx}(i:i+patch_size-1, j:j+patch_size-1)';
    patches_low(:, k) = patch_low(:);
    
    i_hi = (i - 1) * scale_factor + 1;
    j_hi = (j - 1) * scale_factor + 1;
    patch_high = images_high{image_idx}(i_hi:i_hi+patch_size_hi-1,...
        j_hi:j_hi+patch_size_hi-1)';
    patches_high(:, k) = patch_high(:);
end
end

