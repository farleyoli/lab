function [M, idx] = normalized_spectral_clustering_SM (A, k, sigma)
% Implementation of the Normalized Spectral Clustering according to Shi and Malik
% algorithm as specified in the paper "A Tutorial on Spectral Clustering" by Ulrike 
% von Luxburg.
% 
% Inputs: 
%  A: Data matrix, where the data vectors A_i are represented as row vectors. 
%     The size of A is m x n where m is the number of vectors and n their dimension.
%  k: Number of clusters
%  sigma: Parameter for constructing similarity graph (Gaussian similarity function)
%
% Outputs:
%  M: Matrix whose k-th row vector is the representive vector of the k-th cluster.
%  idx: idx(i) in {1, ..., m} indicates which cluster A_i is an element of.
%
% Examples:
%  [M, idx] = unnormalized_spectral_clustering(A,2,1); returns the matrix M where the row
%  vectors represent each cluster and array idx where idx(i) indicates which cluster the
%  i-th data vector is an element of. (two clusters)
   
   %A = standardizeR(A);

   [m,n] = size (A);
   M = zeros(k,n);
   W = zeros(m,m); % similarity graph

   % construct similarity graph (fully connected graph constructed using the Gaussian
   % similarity function)
   for(i = 1:m)
      for(j = 1:m)
         W(i,j) = exp(-(norm(A(i,:) - A(j,:))^2)/(2*sigma*sigma));
      end
   end

   % compute the unnormalized laplacian L
   deg = zeros(1,m);
   for(i = 1:m)
      for(j = 1:m)
         deg(i) = deg(i) + W(i,j);
      end
   end
   D = diag(deg);
   L = D - W;


   % compute the first k eigenvectors V_1, ..., V_k of L
   % corresponding to the k smallest eigenvalues (D_11 is the smallest eigenvalue)
   % in here, we solve the general eigenvalue problem
   V = zeros(n, k); % eigenvalues as columns
   [V,E] = eigs(L, D, k, 'sm');
   E = fliplr(E);
   E = flipud(E);
   V = fliplr(V);

   % for i = 1,...,n, let y_i in R^k be the vector corresponding to the i-th row of V
   for(i = 1:m)
      Y(i,:) = V(i,:);
   end
   

   [Mt, idx] = kmeans(Y, k);

   % calculate the matrix M for the original data 
   for(i = 1:k)
      sum = zeros(1,n);
      n_elem = 0;
      for(j = 1:m)
         if(idx(j) == i)
            sum = sum + A(j,:);
            n_elem = n_elem + 1;
         end
      end
      M(i,:) = sum / n_elem;
   end


end

