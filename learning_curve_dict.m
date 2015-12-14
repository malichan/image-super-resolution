%% set parameters
image_dir = 'dataset/flower';
test_ratio = 0.2;
train_patches = 50000;
test_patches = 200;
scale_factor = 3;
patch_size = 3;
overlap_width = 1;
dict_size = 1024;
hidden_units = 81;
rand_range = 0.01;
learning_rate = 0.1;
max_epochs = 1000;

%% other parameters
image_files = dir(fullfile(image_dir, '*.bmp'));
num_images = size(image_files, 1);
test_size = round(num_images * test_ratio);
train_size = num_images - test_size;
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
indices_test = indices_perm(num_images-test_size+1:end);

%% decompose into patches
[patches_train_high, patches_train_low] = sample_patch_pair(...
    images_high(indices_train), images_low(indices_train),...
    patch_size, scale_factor, train_patches);
[patches_test_high, patches_test_low] = sample_patch_pair(...
    images_high(indices_train), images_low(indices_train),...
    patch_size, scale_factor, test_patches);

%% normalize patches
patches_train_high_norm = normalize_patch(patches_train_high);
patches_train_low_norm = normalize_patch(patches_train_low);

patches_test_high_norm = normalize_patch(patches_test_high);
patches_test_low_norm = normalize_patch(patches_test_low);
%% construct coupled dictionary
training_set_size_list = [2000, 5000, 10000, 20000, 50000];
dict_high=0;
dict_low=0;
squared_error = zeros(length(training_set_size_list));
current_patches_train_high=0;
current_patches_train_low=0;
test_idx = 0;
idx = 0;

for i = 1 : length(training_set_size_list)
    training_set_size = training_set_size_list(i);
    current_patches_train_low = zeros(patch_size*patch_size, training_set_size);
    current_patches_train_high = zeros(patch_size_hi*patch_size_hi, training_set_size);
    
    times = 5;
    if i < length(training_set_size_list)
        times = 5;
    else
        times = 1;
    end
    
    sum_error = 0;
    for t = 1 : times
        
        fprintf('begin %d %d \n',training_set_size, t);
        
        ran = randperm( train_patches);
        for c = 1 : training_set_size
            current_patches_train_high(:,c) = patches_train_high_norm(:,ran(c));
            current_patches_train_low(:,c) = patches_train_low_norm(:,ran(c));
        end
        
        [idx, dict_low] = kmeans(current_patches_train_low', dict_size, 'Display', 'iter', 'MaxIter', 200);
        dict_low = dict_low';
        dict_high = zeros(size(current_patches_train_high, 1), dict_size);
        for c = 1:dict_size
            dict_high(:, c) = mean(current_patches_train_high(:, idx==c), 2);
        end
        
        test_idx = zeros(test_patches, 1);
        for c = 1 : test_patches
            test_idx(c) = 0;
            small = 100000;
            test_patch_low = patches_test_low_norm(:,c);
            
            for d = 1 : dict_size
               %cur_error = mse( test_patch_low, dict_low(:,d));
               cur_error = 0;
               for j = 1 : 9
                  cur_error = cur_error + (test_patch_low(j) - dict_low(j,d)).^ 2; 
               end
               if cur_error < small
                   small = cur_error;
                   test_idx(c) = d;
               end
            end
        end
        
        fprintf('end %d %d \n',training_set_size, t);
        
        for c = 1 : test_patches
            test_patch_high = patches_test_high_norm(:,c);
            sum_error =  sum_error + mse(test_patch_high, dict_high(test_idx(c)) );
        end
        
    end
    sum_error = sum_error / times;
    squared_error(i) = sum_error;
end


