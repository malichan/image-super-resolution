function [ image ] = reconstruct_patch( patches, image_size )
% Reconstruct patches into an image
% Input:  patches - list of patches
%         image_size - size of image matrix
% Output: image - image matrix

height = image_size(1);
width = image_size(2);
patch_size = sqrt(size(patches, 1));
npatch_v = ceil((height - 1) / (patch_size - 1));
npatch_h = ceil((width - 1) / (patch_size - 1));
height_padded = npatch_v * (patch_size - 1) + 1;
width_padded = npatch_h * (patch_size - 1) + 1;

sum = zeros(height_padded, width_padded);
count = zeros(height_padded, width_padded);
for i = 1:npatch_v
    for j = 1:npatch_h
        h_offset = (i - 1) * (patch_size - 1);
        w_offset = (j - 1) * (patch_size - 1);
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

