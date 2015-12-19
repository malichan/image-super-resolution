%% set parameters
image_dir = 'dataset/flower';
train_patches_cd = 10000;
train_patches_nn = 10000;
validation_patches_nn = 2000;
scale_factor = 3;
patch_size = 3;
overlap_width = 1;
dict_size = 1024;
hidden_units = 81;
rand_range = 0.01;
learning_rate = 0.15;
max_epochs = 500;
num_folds = 5;

%% other parameters
image_files = dir(fullfile(image_dir, '*.bmp'));
num_images = size(image_files, 1);
patch_size_hi = patch_size * scale_factor;
overlap_width_hi = overlap_width * scale_factor;
input_units = patch_size * patch_size + patch_size_hi * patch_size_hi;
input_units_nn = patch_size * patch_size;
output_units = patch_size_hi * patch_size_hi;
fold_size = num_images / num_folds;

%% load & downscale images
images_high = cell(1, num_images);
images_low = cell(1, num_images);
for i = 1:num_images
    image = load_grayscale(fullfile(image_dir, image_files(i).name));
    [images_high{i}, images_low{i}] = down_scale(image, scale_factor);
end

%% partition datasets
indices_perm = randperm(num_images);

%% cross validation
error = zeros(2, num_folds);
for k = 1:num_folds
    % set up datasets
    offset = (k - 1) * fold_size;
    indices_train = indices_perm(offset+1:offset+fold_size);
    indices_test = indices_perm(:);
    indices_test(offset+1:offset+fold_size) = [];

    % construct coupled dictionary
    [patches_cd_high, patches_cd_low] = sample_patch_pair(...
        images_high(indices_train), images_low(indices_train),...
        patch_size, scale_factor, train_patches_cd);
    patches_cd_high = normalize_patch(patches_cd_high);
    patches_cd_low = normalize_patch(patches_cd_low);
    [dict_high_1, dict_low_1] = build_dictionary(...
        patches_cd_high, patches_cd_low, dict_size);

    % train neural network 
    [patches_nnt_high, patches_nnt_low] = sample_patch_pair(...
        images_high(indices_train), images_low(indices_train),...
        patch_size, scale_factor, train_patches_nn);
    [patches_nnv_high, patches_nnv_low] = sample_patch_pair(...
        images_high(indices_train), images_low(indices_train),...
        patch_size, scale_factor, validation_patches_nn);
    input_nnt = normalize_patch(patches_nnt_low);
    input_nnt = lookup_dictionary(input_nnt, dict_high_1, dict_low_1);
    input_nnt = [input_nnt; patches_nnt_low];
    input_nnv = normalize_patch(patches_nnv_low);
    input_nnv = lookup_dictionary(input_nnv, dict_high_1, dict_low_1);
    input_nnv = [input_nnv; patches_nnv_low];
    output_nnt = patches_nnt_high;
    output_nnv = patches_nnv_high;

    errors_nnt = zeros(1, max_epochs);
    errors_nnv = zeros(1, max_epochs);
    [weights_in_1, weights_out_1] = initialize_neuralnet(...
        input_units, hidden_units, output_units, rand_range);
    for e = 1:max_epochs
        [weights_in_1, weights_out_1] = train_neuralnet(...
            weights_in_1, weights_out_1, input_nnt, output_nnt, learning_rate);
        pred_nnt = predict_neuralnet(input_nnt, weights_in_1, weights_out_1);
        errors_nnt(e) = sum(sum((pred_nnt - output_nnt) .^ 2, 1), 2) /...
            size(input_nnt, 2);
        pred_nnv = predict_neuralnet(input_nnv, weights_in_1, weights_out_1);
        errors_nnv(e) = sum(sum((pred_nnv - output_nnv) .^ 2, 1), 2) /...
            size(input_nnv, 2);
        fprintf('Epoch %d - Training Error: %f, Validation Error: %f\n',...
            e, errors_nnt(e), errors_nnv(e));

        if e > 1 && errors_nnv(e) > errors_nnv(e - 1)
            break;
        end
    end
    
    % construct coupled dictionary
    [patches_cd_high, patches_cd_low] = sample_patch_pair_alter(...
        images_high(indices_train), images_low(indices_train),...
        patch_size, scale_factor, train_patches_cd);
    patches_cd_high = normalize_patch(patches_cd_high);
    patches_cd_low = normalize_patch(patches_cd_low);
    [dict_high_2, dict_low_2] = build_dictionary(...
        patches_cd_high, patches_cd_low, dict_size);

    % train neural network 
    [patches_nnt_high, patches_nnt_low] = sample_patch_pair(...
        images_high(indices_train), images_low(indices_train),...
        patch_size, scale_factor, train_patches_nn);
    [patches_nnv_high, patches_nnv_low] = sample_patch_pair(...
        images_high(indices_train), images_low(indices_train),...
        patch_size, scale_factor, validation_patches_nn);
    input_nnt = normalize_patch(patches_nnt_low);
    input_nnt = lookup_dictionary(input_nnt, dict_high_2, dict_low_2);
    input_nnt = [input_nnt; patches_nnt_low];
    input_nnv = normalize_patch(patches_nnv_low);
    input_nnv = lookup_dictionary(input_nnv, dict_high_2, dict_low_2);
    input_nnv = [input_nnv; patches_nnv_low];
    output_nnt = patches_nnt_high;
    output_nnv = patches_nnv_high;

    errors_nnt = zeros(1, max_epochs);
    errors_nnv = zeros(1, max_epochs);
    [weights_in_2, weights_out_2] = initialize_neuralnet(...
        input_units, hidden_units, output_units, rand_range);
    for e = 1:max_epochs
        [weights_in_2, weights_out_2] = train_neuralnet(...
            weights_in_2, weights_out_2, input_nnt, output_nnt, learning_rate);
        pred_nnt = predict_neuralnet(input_nnt, weights_in_2, weights_out_2);
        errors_nnt(e) = sum(sum((pred_nnt - output_nnt) .^ 2, 1), 2) /...
            size(input_nnt, 2);
        pred_nnv = predict_neuralnet(input_nnv, weights_in_2, weights_out_2);
        errors_nnv(e) = sum(sum((pred_nnv - output_nnv) .^ 2, 1), 2) /...
            size(input_nnv, 2);
        fprintf('Epoch %d - Training Error: %f, Validation Error: %f\n',...
            e, errors_nnt(e), errors_nnv(e));

        if e > 1 && errors_nnv(e) > errors_nnv(e - 1)
            break;
        end
    end
    
    % test on images
    error_sum_1 = 0;
    error_sum_2 = 0;
    error_count = 0;
    for t = 1:length(indices_test)
        image_test_high = images_high{indices_test(t)};
        image_test_low = images_low{indices_test(t)};
        patches_test_low = decompose_patch(image_test_low,...
            patch_size, overlap_width);
        patches_test_low_norm = normalize_patch(patches_test_low);
        
        patches_test_high_tmp = lookup_dictionary(patches_test_low_norm, dict_high_1, dict_low_1);
        patches_test_high = predict_neuralnet([patches_test_high_tmp; patches_test_low],...
            weights_in_1, weights_out_1);
        image_test_high_sr = reconstruct_patch(patches_test_high,...
            size(image_test_high), overlap_width_hi);
        image_test_high_sr = global_optimize(image_test_high_sr, image_test_low);
        error_sum_1 = error_sum_1 +...
            sum(sum((image_test_high_sr - image_test_high) .^ 2, 1), 2);
        
        patches_test_high_tmp = lookup_dictionary(patches_test_low_norm, dict_high_2, dict_low_2);
        patches_test_high = predict_neuralnet([patches_test_high_tmp; patches_test_low],...
            weights_in_2, weights_out_2);
        image_test_high_sr = reconstruct_patch(patches_test_high,...
            size(image_test_high), overlap_width_hi);
        image_test_high_sr = global_optimize(image_test_high_sr, image_test_low);
        error_sum_2 = error_sum_2 +...
            sum(sum((image_test_high_sr - image_test_high) .^ 2, 1), 2);
        
        error_count = error_count +...
            size(image_test_high, 1) * size(image_test_high, 2);
    end

    error(1, k) = error_sum_1 / error_count;
    error(2, k) = error_sum_2 / error_count;
end

%% show results
average = mean(error, 2);
stddev = std(error, 0, 2);

figure;
bar(1:2, average, 0.5, 'FaceColor', [0.8, 0.8, 0.8]);
hold on;
errorbar(1:2, average, stddev, '.');
set(gca, 'XTick', 1:2, 'XTickLabel',...
    {'Normal Sampling', 'Alternative Sampling'});
title('Contributions of Modules in the Pipeline');
ylabel('Cross Validation Mean Squared Error');
ylim([0.0, 0.0025]);
