%% set parameters
image_dir = 'dataset/flower';
scale_factor = 3;
patch_size = 3;
dict_size = 1024;
train_patches = [2000, 5000, 10000, 20000, 50000];
test_patches = 2000;
repeat_times = 5;

%% other parameters
image_files = dir(fullfile(image_dir, '*.bmp'));
num_images = size(image_files, 1);
patch_size_hi = scale_factor * patch_size;

%% load & downscale images
images_high = cell(1, num_images);
images_low = cell(1, num_images);
for i = 1:num_images
    image = load_grayscale(fullfile(image_dir, image_files(i).name));
    [images_high{i}, images_low{i}] = down_scale(image, scale_factor);
end

%% prepare test patches
[output_test, input_test] = sample_patch_pair(...
    images_high, images_low, patch_size, scale_factor, test_patches);
output_test = normalize_patch(output_test);
input_test = normalize_patch(input_test);

%% construct coupled dictionary
error = zeros(length(train_patches), repeat_times);
for i = 1:length(train_patches)
    for j = 1:repeat_times
        [output_train, input_train] = sample_patch_pair(...
            images_high, images_low, patch_size, scale_factor,...
            train_patches(i));
        output_train = normalize_patch(output_train);
        input_train = normalize_patch(input_train);
        
        [dict_high, dict_low] = build_dictionary(...
            output_train, input_train, dict_size);
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
errorbar(train_patches, average, stddev);
title('Coupled Dictionary Learning Curve');
xlabel('Training-set Size');
ylabel('Test-set Mean Squared Error');
xlim([0, 60000]);
ylim([0.05, 0.35]);