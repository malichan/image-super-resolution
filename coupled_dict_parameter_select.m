%% set parameters
image_dir = 'dataset/flower';
scale_factor = 3;
patch_size = 3;
dict_size = [128, 256, 512, 1024, 2048];
num_patches = 12000;
num_folds = 6;

%% other parameters
image_files = dir(fullfile(image_dir, '*.bmp'));
num_images = size(image_files, 1);
patch_size_hi = scale_factor * patch_size;
fold_size = num_patches / num_folds;

%% load & downscale images
images_high = cell(1, num_images);
images_low = cell(1, num_images);
for i = 1:num_images
    image = load_grayscale(fullfile(image_dir, image_files(i).name));
    [images_high{i}, images_low{i}] = down_scale(image, scale_factor);
end

%% prepare patches
[patches_high, patches_low] = sample_patch_pair(...
    images_high, images_low, patch_size, scale_factor, num_patches);
patches_high = normalize_patch(patches_high);
patches_low = normalize_patch(patches_low);

%% cross validation
error = zeros(length(dict_size), num_folds);
for i = 1:length(dict_size)
    for j = 1:num_folds
        offset = (j - 1) * fold_size;
        input_test = patches_low(:, offset+1:offset+fold_size);
        output_test = patches_high(:, offset+1:offset+fold_size);
        input_train = patches_low;
        input_train(:, offset+1:offset+fold_size) = [];
        output_train = patches_high;
        output_train(:, offset+1:offset+fold_size) = [];
        
        [dict_high, dict_low] = build_dictionary(...
            output_train, input_train, dict_size(i));
        pred_test = lookup_dictionary(...
            input_test, dict_high, dict_low);
        
        error(i, j) = sum(sum((pred_test - output_test) .^ 2, 1), 2)...
            / size(input_test, 2);
    end
end

%% plot figure
average = mean(error, 2);
stddev = std(error, 0, 2);
figure;
errorbar(dict_size, average, stddev);
title('Coupled Dictionary Parameter Selection');
xlabel('Coupled Dictionary Size');
ylabel('Cross-Validation Mean Squared Error');
xlim([0, 2200]);
ylim([0.05, 0.35]);