function fprint_matrix( matrix, file_name )
% write a matrix to text file
% Input:  matrix - matrix
%         file_name - name of text file

file = fopen(file_name, 'w');
[m, n] = size(matrix);
fprintf(file, '%d %d \n', m, n);
fprintf(file, [repmat('%.6f ', 1, n), '\n'], matrix');
fclose(file);

end

