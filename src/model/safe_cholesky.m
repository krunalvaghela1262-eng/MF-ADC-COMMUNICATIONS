function L = safe_cholesky(Sigma)
%SAFE_CHOLESKY  Lower Cholesky factor of a covariance matrix, lifting a ridge if needed.
%
%   L = SAFE_CHOLESKY(Sigma) returns L with Sigma ~= L*L'. The Mahalanobis
%   distance r' * inv(Sigma) * r is then evaluated as ||L \ r||^2, which
%   avoids forming an explicit inverse of an ill-conditioned matrix.

    Sigma = 0.5 * (Sigma + Sigma.');
    [L, flag] = chol(Sigma, 'lower');

    ridge = 1e-8 * mean(diag(Sigma));
    while flag ~= 0 && ridge < 1e3 * mean(diag(Sigma))
        [L, flag] = chol(Sigma + ridge*eye(size(Sigma, 1)), 'lower');
        ridge = ridge * 10;
    end
    assert(flag == 0, 'safe_cholesky: covariance is not positive definite.');
end
