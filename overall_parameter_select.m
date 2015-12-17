%% set parameters
image_dir = 'dataset/flower';
train_patches_cd = 10000;
train_patches_nn = 10000;
validation_patches_nn = 2000;
scale_factor = 3;
patch_size = [2, 3, 4, 5];
dict_size = 1024;
hidden_units = 81;
rand_range = 0.01;
learning_rate = 0.15;
max_epochs = 500;
num_folds = 5;

%% other parameters
image_files = dir(fullfile(image_dir, '*.bmp'));
num_images = size(image_files, 1);
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
error = zeros(length(patch_size), max(patch_size), num_folds);
for i = 1:length(patch_size)
    patch_size_lo = patch_size(i);
    patch_size_hi = scale_factor * patch_size_lo;
    input_units = patch_size_lo * patch_size_lo + patch_size_hi * patch_size_hi;
    output_units = patch_size_hi * patch_size_hi;
    
    for k = 1:num_folds
        % set up datasets
        offset = (k - 1) * fold_size;
        indices_train = indices_perm(offset+1:offset+fold_size);
        indices_test = indices_perm(:);
        indices_test(offset+1:offset+fold_size) = [];

        % construct coupled dictionary
        [patches_cd_high, patches_cd_low] = sample_patch_pair(...
            images_high(indices_train), images_low(indices_train),...
            patch_size_lo, scale_factor, train_patches_cd);
        patches_cd_high = normalize_patch(patches_cd_high);
        patches_cd_low = normalize_patch(patches_cd_low);
        [dict_high, dict_low] = build_dictionary(...
            patches_cd_high, patches_cd_low, dict_size);

        % train neural network
        [patches_nnt_high, patches_nnt_low] = sample_patch_pair(...
            images_high(indices_train), images_low(indices_train),...
            patch_size_lo, scale_factor, train_patches_nn);
        [patches_nnv_high, patches_nnv_low] = sample_patch_pair(...
            images_high(indices_train), images_low(indices_train),...
            patch_size_lo, scale_factor, validation_patches_nn);
        input_nnt = normalize_patch(patches_nnt_low);
        input_nnt = lookup_dictionary(input_nnt, dict_high, dict_low);
        input_nnt = [input_nnt; patches_nnt_low];
        input_nnv = normalize_patch(patches_nnv_low);
        input_nnv = lookup_dictionary(input_nnv, dict_high, dict_low);
        input_nnv = [input_nnv; patches_nnv_low];
        output_nnt = patches_nnt_high;
        output_nnv = patches_nnv_high;

        errors_nnt = zeros(1, max_epochs);
        errors_nnv = zeros(1, max_epochs);
        [weights_in, weights_out] = initialize_neuralnet(...
            input_units, hidden_units, output_units, rand_range);
        for e = 1:max_epochs
            [weights_in, weights_out] = train_neuralnet(...
                weights_in, weights_out, input_nnt, output_nnt, learning_rate);
            pred_nnt = predict_neuralnet(input_nnt, weights_in, weights_out);
            errors_nnt(e) = sum(sum((pred_nnt - output_nnt) .^ 2, 1), 2) /...
                size(input_nnt, 2);
            pred_nnv = predict_neuralnet(input_nnv, weights_in, weights_out);
            errors_nnv(e) = sum(sum((pred_nnv - output_nnv) .^ 2, 1), 2) /...
                size(input_nnv, 2);
            fprintf('Epoch %d - Training Error: %f, Validation Error: %f\n',...
                e, errors_nnt(e), errors_nnv(e));

            if e > 1 && errors_nnv(e) > errors_nnv(e - 1)
                break;
            end
        end
        
        for j = 0:patch_size(i)-1
            overlap_width_lo = j;
            overlap_width_hi = scale_factor * overlap_width_lo;
            
            % test on images
            error_sum = 0;
            error_count = 0;
            for t = 1:length(indices_test)
                image_test_high = images_high{indices_test(t)};
                image_test_low = images_low{indices_test(t)};
                patches_test_low = decompose_patch(image_test_low,...
                    patch_size_lo, overlap_width_lo);
                temp_test = normalize_patch(patches_test_low);
                temp_test = lookup_dictionary(temp_test, dict_high, dict_low);
                temp_test = [temp_test; patches_test_low];
                temp_test = predict_neuralnet(temp_test, weights_in, weights_out);
                temp_test = reconstruct_patch(temp_test,...
                    size(image_test_high), overlap_width_hi);
                image_test_high_sr = global_optimize(temp_test, image_test_low);

                error_sum = error_sum +...
                    sum(sum((image_test_high_sr - image_test_high) .^ 2, 1), 2);
                error_count = error_count +...
                    size(image_test_high, 1) * size(image_test_high, 2);
            end
            
            error(i, j + 1, k) = error_sum / error_count;
        end
    end
end

%% plot figure
average = mean(error, 3);
stddev = std(error, 0, 3);

figure;
bar(0:4, average(4,:), 0.5, 'FaceColor', [0.8, 0.8, 0.8]);
hold on;
errorbar(0:4, average(4,:), stddev(4,:), '.');
title('Overall Parameter Selection (Patch Size = 5)');
xlabel('Overlap Width');
ylabel('Cross Validation Mean Squared Error');
ylim([0.0, 0.0025]);

figure;
bar(2:5, [average(1,2),average(2,3),average(3,4),average(4,5)],...
    0.5, 'FaceColor', [0.8, 0.8, 0.8]);
hold on;
errorbar(2:5, [average(1,2),average(2,3),average(3,4),average(4,5)],...
    [stddev(1,2),stddev(2,3),stddev(3,4),stddev(4,5)], '.');
title('Overall Parameter Selection (Overlap Width = Patch Size - 1)');
xlabel('Patch Size');
ylabel('Cross Validation Mean Squared Error');
ylim([0.0, 0.0025]);