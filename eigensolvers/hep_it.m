function [lambda,x] = hep_it(A, shift, z)
% Inverse Iteration for Hermitian Eigenvalue Problem as described in the textbook
% "Templates for the Solution of Algebraic Eigenvalue Problems A Practical Guide".
% Solves A*x = lambda*x finding the solution in which
% lambda is the closest to shift.
%
% Inputs:
%  z:       initial guess for eigenvector.
%  shift:   shift in the algorithm.
%  A:       hermitian matrix whose eigenvalues are to be 
%           calculated.
%
% Outputs:
%  lambda:  calculated eigenvalue.
%  x:       calculated eigenvector.
%
% Examples:
%  [lambda, x] = hep_it(A); returns eigenpair of the 
%  hermitian matrix A.
%  [lambda, x] = hep_it(A, shift); same as above, but 
%  with predetermined shift.
%  [lambda, x] = hep_it(A, shift, z); same as above, but
%  with initial guess for eigenvector z.

   n = size(A,1); %size of the matrix A

   if (~exist('z'))
      z = rand(n,1);
   end
   if (~exist('shift'))
      shift = 0;
   end

   y = z; %initial guess for eigenvector

   while(true)
      v = y/norm(y);
      y = (A - shift*eye(n))\v;
      theta = v'*y;
      if norm(y - theta*v) <= 10^(-10)*abs(theta)
         break;
      end
   end

   lambda = shift + (1/theta);
   x = y/theta;

end
