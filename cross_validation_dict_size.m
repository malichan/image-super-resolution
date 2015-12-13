%% set parameters
image_dir = 'dataset/flower';
test_ratio = 0.2;
train_patches = 50000;
validation_patches = 10000;
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
% patches_train_high = zeros(patch_size_hi * patch_size_hi, 0);
% patches_train_low = zeros(patch_size * patch_size, 0);
% for i = 1:train_size
%     patches_train_high = [patches_train_high,...
%         decompose_patch(images_high{indices_train(i)}, patch_size_hi, overlap_width_hi)];
%     patches_train_low = [patches_train_low,...
%         decompose_patch(images_low{indices_train(i)}, patch_size, overlap_width)];
% end
% 
% patches_validation_high = zeros(patch_size_hi * patch_size_hi, 0);
% patches_validation_low = zeros(patch_size * patch_size, 0);
% for i = 1:validation_size
%     patches_validation_high = [patches_validation_high,...
%         decompose_patch(images_high{indices_validation(i)}, patch_size_hi, overlap_width_hi)];
%     patches_validation_low = [patches_validation_low,...
%         decompose_patch(images_low{indices_validation(i)}, patch_size, overlap_width)];
% end
[patches_train_high, patches_train_low] = sample_patch_pair(...
    images_high(indices_train), images_low(indices_train),...
    patch_size, scale_factor, train_patches);
[patches_validation_high, patches_validation_low] = sample_patch_pair(...
    images_high(indices_train), images_low(indices_train),...
    patch_size, scale_factor, validation_patches);

%% normalize patches
patches_train_high_norm = normalize_patch(patches_train_high);
patches_train_low_norm = normalize_patch(patches_train_low);

%% construct coupled dictionary

dict_size_list = [128, 256, 512, 1024, 2048];
fold_size = 5;

dict_size_cross_validation_squared_error = zeros(length(dict_size_list), fold_size);

idx;
dict_low;
idx_tuning;

for dict_size_index = 1 : length(dict_size_list)
    dict_size = dict_size_list(dict_size_index);
    for fold = 1 : fold_size
            
            training_size = train_patches * (fold_size - 1) / fold_size;
            tuning_size = train_patches * (1) / fold_size;
            
            training_patches_high = zeros(patch_size_hi*patch_size_hi, training_size);
            training_patches_low = zeros(patch_size*patch_size, training_size);
            tuning_patches_high = zeros(patch_size_hi*patch_size_hi, tuning_size);
            tuning_patches_low = zeros(patch_size*patch_size, tuning_size);
            
            for c = 1 : train_patches
                col = ceil(c/dict_size);
                if mod(c, fold_size)==0
                    tuning_patches_high(:,col) = patches_train_high_norm(:,c);
                    tuning_patches_low(:,col) = patches_train_low_norm(:,c);
                else
                    training_patches_high(:,col) = patches_train_high_norm(:,c);
                    training_patches_low(:,col) = patches_train_low_norm(:,c);
                end
            end
                        
            [idx, dict_low] = kmeans(training_patches_low', dict_size, 'Display', 'iter', 'MaxIter', 200);
            dict_low = dict_low';
            
            dict_high = zeros(patch_size_hi * patch_size_hi, dict_size);
            for i = 1 : dict_size
               dict_high(:,i) = mean( training_patches_high(:,idx==i) ,2); 
            end
            
            idx_tuning = zeros(tuning_size,1);
            for t = 1 : tuning_size
                idx_tuning(t) = 0;
                smallest_dist = 100000000;
                current_patch = tuning_patches_low(:,t);
                for dict_id = 1 : dict_size
                    %dict_error = mean(current_patch-dict_low(:,dict_id));
                    dict_error = 0;
                    for c = 1 : patch_size * patch_size
                       dict_error = dict_error + (current_patch(c) - dict_low(c,dict_id)).^ 2;
                    end
                    if smallest_dist > dict_error
                       smallest_dist = dict_error;
                       idx_tuning(t) = dict_id;
                    end
                end
                
                %disp([t, smallest_dist, idx_tuning(t)]);
            end
            
            %dict_high_tuning = zeros(patch_size*patch_size, dict_size);
            %for i = 1 : dict_size
            %    dict_high_tuning(:,i) = mean( tuning_patches_high(:,idx==i) ,2);
            %end
            
            sum_error = 0;
            for col = 1 : tuning_size
               for row = 1 : patch_size_hi * patch_size_hi
                   sum_error = sum_error + (  tuning_patches_high(row,col) - dict_high(row, idx_tuning(col)) ).^ 2;
               end
            end
            
            dict_size_cross_validation_squared_error(dict_size_index, fold) = sum_error;
    end
end
