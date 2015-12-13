%% visualize the seeds of kmeans in low dictionary

visualize_row = 10;
visualize_col = 20;
visualize_size = visualize_row * visualize_col;

min_value = min(min(dict_low));
max_value = max(max(dict_low));


for i = 1 : visualize_size
    seed = dict_low(:,i);
    seed = reshape(seed, [patch_size, patch_size]);
%    for row = 1 : patch_size
%       for col = 1 : patch_size
%          seed(row ,col) = (seed(row, col) - min_value) / (max_value - min_value) * 255 ;
%       end
%    end
    subplot(visualize_row, visualize_col, i);
    imshow(seed);
end

%% visualize the seeds of kmeans in high dictionary

visualize_row = 5;
visualize_col = 10;
visualize_size = visualize_row * visualize_col;

min_value = min(min(dict_high));
max_value = max(max(dict_high));

figure
for i = 1 : visualize_size
    seed = dict_high(:,i);
    seed = reshape(seed, [patch_size_hi, patch_size_hi]);
%    for row = 1 : patch_size_hi
%       for col = 1 : patch_size_hi
%          seed(row ,col) = (seed(row, col) - min_value) / (max_value - min_value) * 255 ;
%       end
%    end
    subplot(visualize_row, visualize_col, i);
    imshow(seed);
end