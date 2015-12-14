%% set parameters
image_dir = 'dataset/flower';
train_patches = 10000;
scale_factor = 3;
patch_size = 3;
dict_size = 1024;

%% other parameters
image_files = dir(fullfile(image_dir, '*.bmp'));
num_images = size(image_files, 1);
patch_size_hi = patch_size * scale_factor;

%% load & downscale images
images_high = cell(1, num_images);
images_low = cell(1, num_images);
for i = 1:num_images
    image = load_grayscale(fullfile(image_dir, image_files(i).name));
    [images_high{i}, images_low{i}] = down_scale(image, scale_factor);
end

%% sample patches
[patches_train_high, patches_train_low] = sample_patch_pair(...
    images_high, images_low, patch_size, scale_factor, train_patches);

%% normalize patches
patches_train_high_norm = normalize_patch(patches_train_high);
patches_train_low_norm = normalize_patch(patches_train_low);

%% construct coupled dictionary
[dict_high, dict_low] = build_dictionary(...
    patches_train_high_norm, patches_train_low_norm, dict_size);

%% visualize the seeds of kmeans in low dictionary
side = floor(sqrt(dict_size));
dict_low_visualization = zeros(...
    side * (patch_size_hi + 1), side * (patch_size_hi + 1));
for i = 1:side
    for j = 1:side
        h_offset = (i - 1) * (patch_size_hi + 1);
        w_offset = (j - 1) * (patch_size_hi + 1);
        patch = dict_low(:, (i - 1) * side + j) + 0.5;
        patch = reshape(patch, [patch_size patch_size])';
        patch = imresize(patch, scale_factor, 'nearest');
        dict_low_visualization(h_offset+1:h_offset+patch_size_hi,...
            w_offset+1:w_offset+patch_size_hi) = patch;
    end
end
figure;
imshow(dict_low_visualization);

%% visualize the seeds of kmeans in high dictionary
side = floor(sqrt(dict_size));
dict_high_visualization = zeros(...
    side * (patch_size_hi + 1), side * (patch_size_hi + 1));
for i = 1:side
    for j = 1:side
        h_offset = (i - 1) * (patch_size_hi + 1);
        w_offset = (j - 1) * (patch_size_hi + 1);
        patch = dict_high(:, (i - 1) * side + j) + 0.5;
        patch = reshape(patch, [patch_size_hi patch_size_hi])';
        dict_high_visualization(h_offset+1:h_offset+patch_size_hi,...
            w_offset+1:w_offset+patch_size_hi) = patch;
    end
end
figure;
imshow(dict_high_visualization);
