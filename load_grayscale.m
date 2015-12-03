function [ image ] = load_grayscale( file_name )
% Load an image and convert it to grayscale (0.0 - 1.0)
% Input:  file_name - image file name
% Output: image - image matrix

image = imread(file_name);
image = rgb2gray(image);
image = double(image) / 255;

end

