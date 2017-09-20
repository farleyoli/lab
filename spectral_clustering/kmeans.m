function [M, idx] = kmeans(A, k)
% Implementation of the k-means algorithm as described in the book "初めてのパターン認識"
% by Yuzo Hirai.
%
% Inputs: 
%  A: Data matrix, where the data vectors A_i are represented as row vectors. 
%     The size of A is m x n where m is the number of vectors and n their dimension.
%  k: Number of clusters
%
% Outputs:
%  M: Matrix where the k-th row vector is the representive vector of the k-th cluster.
%  idx: idx(i) in {1, ..., m} indicates which cluster A_i is an element of.
%
% Examples:
%  [M, idx] = kmeans(A,k); returns the matrix M where the row vectors represent
%  each cluster and array idx where idx(i) indicates which cluster the i-th data vector
%  is an element of.

   %A = standardizeR(A);

   [m,n] = size(A);
   % Q_ij is 1 if A_i is an element of the j-th cluster; otherwise, it is 0
   Q = zeros(m,k);
   newQ = zeros(m,k);
   M = zeros(k, n);

   % Initialization: distribute the data vectors to clusters randomly and compute the
   % mean vectors of each cluster
   for (i = 1:m)
     Q(i, randi(k)) = 1; 
   end
   for(i = 1:k)
      sum = zeros(1,n);
      n_elem = 0;
      for(j = 1:m)
         if(Q(j,i) == 1)
            sum = sum + A(j,:);
            n_elem = n_elem + 1;
         end
      end
      M(i,:) = sum / n_elem;
   end

   while (true)
      % Optimization for Q
      newQ = zeros(m,k);
      for (i = 1:m)
         kt = m+1;
         min = 1/0; % Infinity
         for (j = 1:k)
            if (norm(A(i,:) - M(j,:)) < min)
               min = norm(A(i,:) - M(j,:));
               kt = j;
            end
         end
         newQ(i,kt) = 1;
      end

      % Optimization for M
      for(i = 1:k)
         sum = zeros(1,n);
         n_elem = 0;
         for(j = 1:m)
            if(newQ(j,i) == 1)
               sum = sum + A(j,:);
               n_elem = n_elem + 1;
            end
         end
         M(i,:) = sum / n_elem;
      end

      % Test for termination condition (no change)
      if(Q == newQ)
         break;
      else
         Q = newQ;
      end
   end

   % Compute idx from Q
   for(i = 1:m)
      if (Q(i,1) == 1)
         idx(i) = 1;
      elseif (Q(i,2) == 1)
         idx(i) = 2;
      else 
         idx(i) = 3;
   end
end
