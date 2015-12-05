%% set parameters
image_dir = 'dataset/flower'; 
validation_ratio = 0.2;
test_ratio = 0.2;
scale_factor = 3;
patch_size = 3;
overlap_width = 1;
dict_size = 1024;
hidden_units = 81;
rand_range = 0.01;
learning_rate = 0.1;
max_epochs = 500;

%% other parameters
image_files = dir(fullfile(image_dir, '*.bmp'));
num_images = size(image_files, 1);
validation_size = round(num_images * validation_ratio);
test_size = round(num_images * test_ratio);
train_size = num_images - validation_size - test_size;
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

%% partition datasets
indices_perm = randperm(num_images);
indices_train = indices_perm(1:train_size);
indices_validation = indices_perm(train_size+1:num_images-test_size);
indices_test = indices_perm(num_images-test_size+1:end);

%% decompose into patches
patches_train_high = zeros(patch_size_hi * patch_size_hi, 0);
patches_train_low = zeros(patch_size * patch_size, 0);
for i = 1:train_size
    patches_train_high = [patches_train_high,...
        decompose_patch(images_high{indices_train(i)}, patch_size_hi, overlap_width_hi)];
    patches_train_low = [patches_train_low,...
        decompose_patch(images_low{indices_train(i)}, patch_size, overlap_width)];
end

patches_validation_high = zeros(patch_size_hi * patch_size_hi, 0);
patches_validation_low = zeros(patch_size * patch_size, 0);
for i = 1:validation_size
    patches_validation_high = [patches_validation_high,...
        decompose_patch(images_high{indices_validation(i)}, patch_size_hi, overlap_width_hi)];
    patches_validation_low = [patches_validation_low,...
        decompose_patch(images_low{indices_validation(i)}, patch_size, overlap_width)];
end

%% normalize patches
patches_train_high_norm = normalize_patch(patches_train_high);
patches_train_low_norm = normalize_patch(patches_train_low);

patches_validation_high_norm = normalize_patch(patches_validation_high);
patches_validation_low_norm = normalize_patch(patches_validation_low);

%% construct coupled dictionary
[dict_high, dict_low] = build_dictionary(...
    [patches_train_high_norm, patches_validation_high_norm],...
    [patches_train_low_norm, patches_validation_low_norm], dict_size);

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
    
    if i > 1 && errors_validation(i) > errors_validation(i - 1)
        break;
    end
end

%% test phase
for i = 1:test_size
    image_test_high = images_high{indices_test(i)};
    image_test_low = images_low{indices_test(i)};
    patches_test_high = decompose_patch(image_test_high, patch_size_hi, overlap_width_hi);
    patches_test_low = decompose_patch(image_test_low, patch_size, overlap_width);
    patches_test_low_norm = normalize_patch(patches_test_low);
    patches_test_high_tmp = lookup_dictionary(patches_test_low_norm, dict_high, dict_low);
    patches_test_high_est = predict_neuralnet([patches_test_high_tmp; patches_test_low], weights_in, weights_out);
    image_test_high_est = reconstruct_patch(patches_test_high_est, size(image_test_high), overlap_width_hi);
    image_test_high_opt = global_optimize(image_test_high_est, image_test_low);
    
    image_test_high_baseline = imresize(image_test_low, scale_factor, 'bicubic');
    imshow([image_test_high, image_test_high_baseline, image_test_high_opt]);
    error_baseline = sum(sum((image_test_high_baseline - image_test_high) .^ 2, 1), 2) / (size(image_test_high, 1) * size(image_test_high, 2));
    error_superres = sum(sum((image_test_high_opt - image_test_high) .^ 2, 1), 2) / (size(image_test_high, 1) * size(image_test_high, 2));
    fprintf('Image %d\n', indices_test(i));
    fprintf('Error of bicubic interpolation: %f\n', error_baseline);
    fprintf('Error of our proposed approach: %f\n', error_superres);
    pause;
end