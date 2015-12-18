%% set parameters
image_dir = 'dataset/flower';
scale_factor = 3;
patch_size = 3;
dict_size = 1024;
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
error = zeros(3, num_folds);
for j = 1:num_folds
    offset = (j - 1) * fold_size;
    input_test = patches_low(:, offset+1:offset+fold_size);
    output_test = patches_high(:, offset+1:offset+fold_size);
    input_train = patches_low;
    input_train(:, offset+1:offset+fold_size) = [];
    output_train = patches_high;
    output_train(:, offset+1:offset+fold_size) = [];

    [dict_high, dict_low] = build_dictionary(...
        output_train, input_train, dict_size);
    
    pred_test = lookup_dictionary(...
        input_test, dict_high, dict_low);
    error(1, j) = sum(sum((pred_test - output_test) .^ 2, 1), 2)...
        / size(input_test, 2);
    
    pred_test = lookup_dictionary_in_sparse_coding(...
        input_test, dict_high, dict_low);
    error(2, j) = sum(sum((pred_test - output_test) .^ 2, 1), 2)...
        / size(input_test, 2);
    
    pred_test = lookup_dictionary_in_LLC(...
        input_test, dict_high, dict_low);
    error(3, j) = sum(sum((pred_test - output_test) .^ 2, 1), 2)...
        / size(input_test, 2);
end

%% plot figure
average = mean(error, 2);
stddev = std(error, 0, 2);
figure;
bar(1:3, average, 0.5, 'FaceColor', [0.8, 0.8, 0.8]);
hold on;
errorbar(1:3, average, stddev, '.');
set(gca, 'XTick', 1:4, 'XTickLabel',...
    {'VQ', 'SC', 'LLC'});
title('Coupled Dictionary Prediction Methods');
ylabel('Cross Validation Mean Squared Error');
ylim([0.05, 0.35]);