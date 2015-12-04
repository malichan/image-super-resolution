function [ image_high_opt ] = global_optimize( image_high, image_low )
% Enforce global contraints on the image
% Input:  image_high - high-resolution image matrix
%         image_low - low-resolution image matrix
% Output: image_high_opt - high-resolution image matrix after optimization

[height_high, width_high] = size(image_high);
[height_low, width_low] = size(image_low);
scale_factor = height_high / height_low;

image_high_opt = zeros(height_high, width_high);
for h = 1:height_low
    for w = 1:width_low
        h_offset = (h - 1) * scale_factor;
        w_offset = (w - 1) * scale_factor;
        px_high = image_high(h_offset+1:h_offset+scale_factor,...
            w_offset+1:w_offset+scale_factor);
        px_low = image_low(h, w);
        
        error = px_low - mean(mean(px_high, 1), 2);
        px_high = px_high + error;
        
        image_high_opt(h_offset+1:h_offset+scale_factor,...
            w_offset+1:w_offset+scale_factor) = px_high;
    end
end

end

