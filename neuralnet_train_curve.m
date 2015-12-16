%% set parameters
image_dir = 'dataset/flower';
train_patches = 10000;
validation_patches = 2000;
scale_factor = 3;
patch_size = 3;
dict_size = 1024;
hidden_units = 81;
rand_range = 0.01;
learning_rate = 0.1;
max_epochs = 500;
train_patches_dict = 10000;

%% other parameters
image_files = dir(fullfile(image_dir, '*.bmp'));
num_images = size(image_files, 1);
patch_size_hi = patch_size * scale_factor;
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

%% sample patches
[patches_train_high, patches_train_low] = sample_patch_pair(...
    images_high, images_low, patch_size, scale_factor, train_patches);
[patches_validation_high, patches_validation_low] = sample_patch_pair(...
    images_high, images_low, patch_size, scale_factor, validation_patches);

%% normalize patches
patches_train_high_norm = normalize_patch(patches_train_high);
patches_train_low_norm = normalize_patch(patches_train_low);

patches_validation_high_norm = normalize_patch(patches_validation_high);
patches_validation_low_norm = normalize_patch(patches_validation_low);

%% lookup in coupled dictionary
patches_train_high_tmp = lookup_dictionary(patches_train_low_norm, dict_high, dict_low);
patches_validation_high_tmp = lookup_dictionary(patches_validation_low_norm, dict_high, dict_low);

%% train neural network
input_train = [patches_train_high_tmp; patches_train_low];
output_train = patches_train_high;
input_validation = [patches_validation_high_tmp; patches_validation_low];
output_validation = patches_validation_high;

errors_train = zeros(1, max_epochs);
errors_validation = zeros(1, max_epochs);

[weights_in, weights_out] = initialize_neuralnet(input_units, hidden_units, output_units, rand_range);
for i = 1:max_epochs
    [weights_in, weights_out] = train_neuralnet(weights_in, weights_out, input_train, output_train, learning_rate);
    pred_train = predict_neuralnet(input_train, weights_in, weights_out);
    errors_train(i) = sum(sum((pred_train - output_train) .^ 2, 1), 2) / size(input_train, 2);
    pred_validation = predict_neuralnet(input_validation, weights_in, weights_out);
    errors_validation(i) = sum(sum((pred_validation - output_validation) .^ 2, 1), 2) / size(input_validation, 2);
    fprintf('Epoch %d - Training Error: %f, Validation Error: %f\n', i, errors_train(i), errors_validation(i));
end

%% plot figure
figure;
hold on;
plot(1:max_epochs, errors_train);
plot(1:max_epochs, errors_validation);
title('Neural Network Training Curve');
xlabel('Training Epochs');
ylabel('Mean Squared Error');
legend('Training-set', 'Validation-set');
ylim([0.05, 0.35]);
xlim([0, 500]);
for i = 1:max_epochs
    if i > 1 && errors_validation(i) > errors_validation(i - 1)
        break;
    end
end
fprintf('Training should stop at epoch %d.\n', i);