function [S] = standardizeR (M)
% This function normalizes the matrix M in such a way that the row vectors end up with mean 0,
% standard deviation 1.

   [m,n] = size(M); % m row vectors of size n
   S = zeros(m,n); % standardized matrix

   % subtract each row by mean and divide by std 
   for i = 1:m
      S(i,:) = (M(i,:) - mean(M(i,:)) * ones(1,n)) ./ std(M(i,:));
   end

end
