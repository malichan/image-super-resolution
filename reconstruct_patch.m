function [ image ] = reconstruct_patch( patches, image_size, overlap_width )
% Reconstruct patches into an image
% Input:  patches - list of patches
%         image_size - size of image matrix
%         overlap_width - overlap width
% Output: image - image matrix

height = image_size(1);
width = image_size(2);
patch_size = sqrt(size(patches, 1));
npatch_v = ceil((height - overlap_width) / (patch_size - overlap_width));
npatch_h = ceil((width - overlap_width) / (patch_size - overlap_width));
height_padded = npatch_v * (patch_size - overlap_width) + overlap_width;
width_padded = npatch_h * (patch_size - overlap_width) + overlap_width;

sum = zeros(height_padded, width_padded);
count = zeros(height_padded, width_padded);
for i = 1:npatch_v
    for j = 1:npatch_h
        h_offset = (i - 1) * (patch_size - overlap_width);
        w_offset = (j - 1) * (patch_size - overlap_width);
        patch = reshape(patches(:, (i - 1) * npatch_h + j),...
            [patch_size, patch_size]);
        for h = 1:patch_size
            for w = 1:patch_size
                hi = h_offset + h;
                wj = w_offset + w;
                sum(hi, wj) = sum(hi, wj) + patch(h, w);
                count(hi, wj) = count(hi, wj) + 1;
            end
        end
    end
end

image = sum ./ count;
image = image(1:height, 1:width);

end

