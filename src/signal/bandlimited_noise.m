function w_cont = bandlimited_noise(t_d, W, sigma_lpf)
%BANDLIMITED_NOISE  Channel noise after the receiver's ideal low-pass filter.
%
%   w_cont = BANDLIMITED_NOISE(t_d, W, sigma_lpf) draws white Gaussian
%   noise on the dense grid t_d and passes it through a (near) brick-wall
%   low-pass filter with cut-off W. The white-noise level is chosen so the
%   in-band standard deviation after filtering equals sigma_lpf:
%
%       sigma_white^2 * (2W * Td) = sigma_lpf^2
%
%   The result is a correlated, bandlimited Gaussian process with
%   autocorrelation R_w(tau) = sigma_lpf^2 * sinc(2W tau).
%
%   Paper: Section 2 (front-end LPF acts as the matched filter for sinc
%   pulses and reshapes wideband noise eta(t) into w(t)).
%
%   Requires the Signal Processing Toolbox (lowpass).

    if sigma_lpf == 0
        w_cont = zeros(numel(t_d), 1);
        return;
    end

    Td          = t_d(2) - t_d(1);
    Fs          = 1 / Td;
    sigma_white = sqrt(sigma_lpf^2 / (2*W*Td));
    white       = sigma_white * randn(numel(t_d), 1);

    w_cont = lowpass(white, W, Fs, ImpulseResponse="iir", Steepness=0.9999);
end
