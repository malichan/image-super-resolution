function [ image_high, image_low ] = down_scale( image, scale_factor )
% Down scale an image (and crop it when needed)
% Input:  image - image matrix
%         scale_factor - scale factor
% Output: image_high - high-resolution image matrix
%         image_low - low-resolution image matrix

[height, width] = size(image);
height_low = floor(height / scale_factor);
height_high = height_low * scale_factor;
width_low = floor(width / scale_factor);
width_high = width_low * scale_factor;

image_high = image(1:height_high, 1:width_high);

image_low = zeros(height_low, width_low);
for h = 1:height_low
    for w = 1:width_low
        h_offset = (h - 1) * scale_factor;
        w_offset = (w - 1) * scale_factor;
        px_high = image_high(h_offset+1:h_offset+scale_factor,...
            w_offset+1:w_offset+scale_factor);
        px_low = mean(mean(px_high, 1), 2);
        image_low(h, w) = px_low;
    end
end

end

