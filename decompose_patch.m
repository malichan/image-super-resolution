function [ patches ] = decompose_patch( image, patch_size, overlap_width )
% Decompose an image into patches
% Input:  image - image matrix
%         patch_size - patch size
%         overlap_width - overlap width
% Output: patches - list of patches

[height, width] = size(image);
npatch_v = ceil((height - overlap_width) / (patch_size - overlap_width));
npatch_h = ceil((width - overlap_width) / (patch_size - overlap_width));
height_padded = npatch_v * (patch_size - overlap_width) + overlap_width;
width_padded = npatch_h * (patch_size - overlap_width) + overlap_width;

image_padded = zeros(height_padded, width_padded);
image_padded(1:height, 1:width) = image;
for i = height+1:height_padded
    image_padded(i, :) = image_padded(height * 2 - i + 1, :);
end
for j = width+1:width_padded
    image_padded(:, j) = image_padded(:, width * 2 - j + 1);
end

patches = zeros(patch_size * patch_size, npatch_v * npatch_h);
for i = 1:npatch_v
    for j = 1:npatch_h
        h_offset = (i - 1) * (patch_size - overlap_width);
        w_offset = (j - 1) * (patch_size - overlap_width);
        patch = image_padded(h_offset+1:h_offset+patch_size,...
            w_offset+1:w_offset+patch_size)';
        patches(:, (i - 1) * npatch_h + j) = patch(:);
    end
end

end