%% set parameters
image_dir = 'dataset/flower';
scale_factor = 3;
patch_size = 3;
dict_size = 1024;
hidden_units = 81;
rand_range = 0.01;
learning_rate = 0.2;
max_epochs = 500;
train_patches = [2000, 5000, 10000, 20000, 50000];
test_patches = 2000;
train_patches_dict = 10000;
repeat_times = 5;

%% other parameters
image_files = dir(fullfile(image_dir, '*.bmp'));
num_images = size(image_files, 1);
patch_size_hi = scale_factor * patch_size;
input_units = patch_size * patch_size + patch_size_hi * patch_size_hi;
output_units = patch_size_hi * patch_size_hi;

%% load & downscale images
images_high = cell(1, num_images);
images_low = cell(1, num_images);
for i = 1:num_images
    image = load_grayscale(fullfile(image_dir, image_files(i).name));
    [images_high{i}, images_low{i}] = down_scale(image, scale_factor);
end

%% construct coupled dictionary
[patches_dict_high, patches_dict_low] = sample_patch_pair(...
    images_high, images_low, patch_size, scale_factor, train_patches_dict);
patches_dict_high = normalize_patch(patches_dict_high);
patches_dict_low = normalize_patch(patches_dict_low);
[dict_high, dict_low] = build_dictionary(...
    patches_dict_high, patches_dict_low, dict_size);

%% prepare test patches
[patches_test_high, patches_test_low] = sample_patch_pair(...
    images_high, images_low, patch_size, scale_factor, test_patches);
patches_test_low_norm = normalize_patch(patches_test_low);
patches_test_high_tmp = lookup_dictionary(patches_test_low_norm, dict_high, dict_low);
output_test = patches_test_high;
input_test = [patches_test_high_tmp; patches_test_low];

%% train neuralnet
error = zeros(length(train_patches), repeat_times);
for i = 1:length(train_patches)
    for j = 1:repeat_times
        [patches_train_high, patches_train_low] = sample_patch_pair(...
            images_high, images_low, patch_size, scale_factor,...
            train_patches(i));
        patches_train_low_norm = normalize_patch(patches_train_low);
        patches_train_high_tmp = lookup_dictionary(patches_train_low_norm, dict_high, dict_low);
        output_train = patches_train_high;
        input_train = [patches_train_high_tmp; patches_train_low];
        
        errors_train = zeros(1, max_epochs);
        errors_test = zeros(1, max_epochs);

        [weights_in, weights_out] = initialize_neuralnet(input_units, hidden_units, output_units, rand_range);
        for e = 1:max_epochs
            [weights_in, weights_out] = train_neuralnet(weights_in, weights_out, input_train, output_train, learning_rate);
            pred_train = predict_neuralnet(input_train, weights_in, weights_out);
            errors_train(e) = sum(sum((pred_train - output_train) .^ 2, 1), 2) / size(input_train, 2);
            pred_test = predict_neuralnet(input_test, weights_in, weights_out);
            errors_test(e) = sum(sum((pred_test - output_test) .^ 2, 1), 2) / size(input_test, 2);
            fprintf('Epoch %d - Training Error: %f, Validation Error: %f\n', e, errors_train(e), errors_test(e));

            if e > 1 && errors_test(e) > errors_test(e - 1)
                break;
            end
        end
        
        error(i, j) = errors_test(e);
    end
end

%% plot figure
average = mean(error, 2);
stddev = std(error, 0, 2);
figure;
errorbar(train_patches, average, stddev);
title('Neural Network Learning Curve');
xlabel('Training-set Size');
ylabel('Test-set Mean Squared Error');
xlim([0, 60000]);
ylim([0.05, 0.35]);