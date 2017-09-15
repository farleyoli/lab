function [lambda, x] = hep_pm(A, z)
% Power Method for Hermitian Eigenvalue Problem as described in the textbook
% "Templates for the Solution of Algebraic Eigenvalue Problems A Practical Guide".
% Solves A*x = lambda*x for a hermitian matrix A.
%
% Inputs:
%  A: hermitian square matrix whose eigenpairs
%     we are going to calculate.
%  z: initial guess of the eigenvector. in case it is not
%     specified, a random vector is used.
%
%
% Outputs:
%  lambda: converged eigenvalue (largest in magnitude).
%  x:      eigenvector corresponding to lambda
%          whose magnitude is 1.
%
%
% Examples:
%  [lambda, x] = hep_pm(A); returns eigenpair of hermitian 
%  matrix A.
%
%  [lambda, x] = hep_pm(A,z); same as above, but uses z as a
%  initial guess for eigenvector.

   n = size(A,1); %size of the matrix A

   if (~exist('z'))
      z = rand(n,1);
   end

   y = z; %initial guess for eigenvector

   while(true)
      v = y/norm(y); %guess for the eigenvector
      y = A*v;
      theta = v'*y; %guess for the eigenvalue
      if norm(y - theta*v) <= eps*abs(theta)
         break;
      end
   end
   x = v;
   lambda = theta;
end
