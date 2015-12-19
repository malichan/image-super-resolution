%% set parameters
image_dir = 'dataset/anime';
train_patches_cd = 20000;
train_patches_nn = 20000;
validation_patches_nn = 4000;
scale_factor = 3;
patch_size = 3;
overlap_width = 2;
dict_size = 512;
hidden_units = 27;
rand_range = 0.01;
learning_rate = 0.1;
max_epochs = 500;
train_size = 80;
test_size = 20;
indices_train = [1:24, 31:54, 61:92];
indices_test = [25:30, 55:60, 93:100];

%% other parameters
image_files = dir(fullfile(image_dir, '*.bmp'));
num_images = size(image_files, 1);
patch_size_hi = scale_factor * patch_size;
overlap_width_hi = scale_factor * overlap_width;
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
    images_high(indices_train), images_low(indices_train),...
    patch_size, scale_factor, train_patches_cd);
patches_dict_high = normalize_patch(patches_dict_high);
patches_dict_low = normalize_patch(patches_dict_low);
[dict_high, dict_low] = build_dictionary(...
    patches_dict_high, patches_dict_low, dict_size);

%% train neural network
[patches_train_high, patches_train_low] = sample_patch_pair(...
    images_high(indices_train), images_low(indices_train),...
    patch_size, scale_factor, train_patches_nn);
[patches_validation_high, patches_validation_low] = sample_patch_pair(...
    images_high(indices_train), images_low(indices_train),...
    patch_size, scale_factor, validation_patches_nn);
patches_train_low_norm = normalize_patch(patches_train_low);
patches_validation_low_norm = normalize_patch(patches_validation_low);
patches_train_high_tmp = lookup_dictionary_in_LLC(patches_train_low_norm, dict_high, dict_low);
patches_validation_high_tmp = lookup_dictionary_in_LLC(patches_validation_low_norm, dict_high, dict_low);

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
    
    if i > 1 && errors_validation(i) > errors_validation(i - 1)
        break;
    end
end

%% test phase
error = zeros(2, test_size);
for i = 1:test_size
    image_test_high = images_high{indices_test(i)};
    image_test_low = images_low{indices_test(i)};
    patches_test_low = decompose_patch(image_test_low, patch_size, overlap_width);
    patches_test_low_norm = normalize_patch(patches_test_low);
    patches_test_high_tmp = lookup_dictionary_in_LLC(patches_test_low_norm, dict_high, dict_low);
    patches_test_high_est = predict_neuralnet([patches_test_high_tmp; patches_test_low], weights_in, weights_out);
    image_test_high_est = reconstruct_patch(patches_test_high_est, size(image_test_high), overlap_width_hi);
    image_test_high_opt = global_optimize(image_test_high_est, image_test_low);
    
    image_test_high_baseline = imresize(image_test_low, scale_factor, 'bicubic');
%     imshow([image_test_high, image_test_high_baseline, image_test_high_opt]);
    error(1, i) = sum(sum((image_test_high_baseline - image_test_high) .^ 2, 1), 2) / (size(image_test_high, 1) * size(image_test_high, 2));
    error(2, i) = sum(sum((image_test_high_opt - image_test_high) .^ 2, 1), 2) / (size(image_test_high, 1) * size(image_test_high, 2));
    fprintf('Image %d\n', indices_test(i));
    fprintf('Error of bicubic interpolation: %f\n', error(1, i));
    fprintf('Error of our proposed approach: %f\n', error(2, i));
%     pause;
end