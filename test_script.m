%% set parameters
scale_factor = 3;
patch_size = 3;
overlap_width = 1;
dict_size = 256;
patch_size_hi = scale_factor * patch_size;
overlap_width_hi = scale_factor * overlap_width;

%% load images
image_train = load_grayscale('dataset/flower/flower01.bmp');
image_test = load_grayscale('dataset/flower/flower50.bmp');

%% downscale images
[image_train_high, image_train_low] = down_scale(image_train, scale_factor);
[image_test_high, image_test_low] = down_scale(image_test, scale_factor);

%% decompose into patches
patches_train_high = decompose_patch(image_train_high, patch_size_hi, overlap_width_hi);
patches_train_low = decompose_patch(image_train_low, patch_size, overlap_width);
patches_test_low = decompose_patch(image_test_low, patch_size, overlap_width);

%% normalize patches
patches_train_high_norm = normalize_patch(patches_train_high);
patches_train_low_norm = normalize_patch(patches_train_low);
patches_test_low_norm = normalize_patch(patches_test_low);

%% construct coupled dictionary
[dict_high, dict_low] = build_dictionary(patches_train_high_norm, patches_train_low_norm, dict_size);

%% lookup in coupled dictionary
patches_test_high_tmp = lookup_dictionary(patches_test_low_norm, dict_high, dict_low);

%% estimate hi-res patches (to be replaced by neural net) 
patches_test_high_est = patches_test_high_tmp + repmat(mean(patches_test_low, 1), size(dict_high, 1), 1);

%% reconstruct from patches
image_test_high_est = reconstruct_patch(patches_test_high_est, size(image_test_high), overlap_width_hi);

%% enforce global contraints
image_test_high_opt = global_optimize(image_test_high_est, image_test_low);

%% show results
image_test_high_bicubic = imresize(image_test_low, scale_factor, 'nearest');
imshow([image_test_high, image_test_high_bicubic, image_test_high_opt]);

error_baseline = sum(sum((image_test_high_bicubic - image_test_high) .^ 2, 1), 2);
error_superres = sum(sum((image_test_high_opt - image_test_high) .^ 2, 1), 2);
disp(['Error of baseline:   ', num2str(error_baseline)]);
disp(['Error of our method: ', num2str(error_superres)]);