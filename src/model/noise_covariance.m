function Sigma_T = noise_covariance(t_s, W, lambda, sigma_lpf, sigma_q)
%NOISE_COVARIANCE  Total noise covariance  Sigma_T = Sigma_Y + sigma_q^2 I  used by MHBL.
%
%   Sigma_T = NOISE_COVARIANCE(t_s, W, lambda, sigma_lpf, sigma_q) builds
%
%       [Sigma_Y]_{ij} = R_Y(t_i - t_j) = sum_m A_{2m+1} rho(t_i - t_j)^(2m+1)
%
%   (the folded channel-noise covariance, see folded_noise_coefficients.m)
%   and adds the post-fold quantization noise, which is independent and
%   approximately white with variance sigma_q^2 = Delta^2/12.
%
%   For lambda/sigma_lpf >= 3 the leading term dominates and
%   Sigma_Y ~= sigma_lpf^2 * sinc(2W tau), i.e. Sigma_T ~= Sigma_w + sigma_q^2 I
%   (paper eq. 10). Using the full series keeps the metric accurate
%   closer to the folding boundary.
%
%   A small relative ridge (1e-6 of the mean diagonal) is added because a
%   sinc-correlation matrix becomes ill-conditioned at high OF.

    t_s      = t_s(:);
    N        = numel(t_s);
    A_coeffs = folded_noise_coefficients(lambda, sigma_lpf);

    rho     = sinc(2*W*(t_s - t_s.'));
    Sigma_Y = zeros(N);
    for m = 0:numel(A_coeffs)-1
        Sigma_Y = Sigma_Y + A_coeffs(m+1) * rho.^(2*m + 1);
    end

    Sigma_T = Sigma_Y + sigma_q^2 * eye(N);
    Sigma_T = 0.5 * (Sigma_T + Sigma_T.');
    Sigma_T = Sigma_T + 1e-6 * mean(diag(Sigma_T)) * eye(N);
end
