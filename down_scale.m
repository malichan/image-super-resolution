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
        sum = 0;
        for i = 1:scale_factor
            for j = 1:scale_factor
                hi = (h - 1) * scale_factor + i;
                wj = (w - 1) * scale_factor + j;
                sum = sum + image_high(hi, wj);
            end
        end
        image_low(h, w) = sum / (scale_factor * scale_factor);
    end
end

end

