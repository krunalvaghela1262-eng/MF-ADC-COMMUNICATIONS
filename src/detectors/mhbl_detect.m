function [a_hat, best_idx, dist] = mhbl_detect(y_lambda, HA, A, L, lambda)
%MHBL_DETECT  Exhaustive Mahalanobis (MHBL) ML symbol detector on folded samples.
%
%   a_hat = MHBL_DETECT(y_lambda, HA, A, L, lambda) returns
%
%       a_hat = argmin_{a in S^Ns}  r(a)' * inv(Sigma_T) * r(a),
%       r(a)  = M_lambda( y_lambda - H a )                   (paper eq. 11)
%
%   Inputs
%       y_lambda  N x 1     folded + quantized ADC samples
%       HA        N x K     H*A, noiseless samples of every candidate
%       A         Ns x K    candidate symbol vectors (enumerate_sequences)
%       L         N x N     lower Cholesky factor of Sigma_T (noise_covariance)
%       lambda              folding threshold
%
%   Why no unfolding is needed (wrap cancellation, paper eq. 6-8):
%   y_lambda - H a_true = w + 2*lambda*z + q, and M_lambda removes the
%   unknown integer term 2*lambda*z exactly. So for the true hypothesis
%   the residual is just the (small) noise, while a wrong hypothesis
%   leaves a large, structured residual. No estimate of z is ever formed.
%
%   Pass L = eye(N) to get the plain Euclidean-on-folded-residual
%   detector (ablation: ignores noise correlation).
%
%   Complexity: O(N^2 |S|^Ns) -- use mhbl_block_detect for long sequences.

    R    = modulo_fold(y_lambda - HA, lambda);   % N x K wrapped residuals
    Z    = L \ R;                                % whitening
    dist = sum(Z.^2, 1);                         % Mahalanobis distance per candidate
    [~, best_idx] = min(dist);
    a_hat = A(:, best_idx);
end
