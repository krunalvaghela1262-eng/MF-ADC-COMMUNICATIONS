function y = modulo_fold(x, lambda)
%MODULO_FOLD  Centred modulo operator  M_lambda(x) = mod(x + lambda, 2*lambda) - lambda.
%
%   y = MODULO_FOLD(x, lambda) folds every entry of x into [-lambda, lambda).
%   Equivalently  x = M_lambda(x) + 2*lambda*z  for an integer folding
%   vector z. Two properties used throughout the detector:
%
%       M_lambda(x + 2*lambda*z) = M_lambda(x)       (2*lambda*Z is the kernel)
%       M_lambda(x) = x            for x in [-lambda, lambda)
%
%   Paper: Section 2, eq. (1)-(2), and the wrap-cancellation step (8).

    y = mod(x + lambda, 2*lambda) - lambda;
end
