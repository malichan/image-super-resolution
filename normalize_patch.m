function [ patches_norm ] = normalize_patch( patches )
% Subtract each patch by its mean
% Input:  patches - list of patches
% Output: patches_norm - list of normalized patches

patch_length = size(patches, 1);
patches_norm = patches - repmat(mean(patches, 1), patch_length, 1);

end

