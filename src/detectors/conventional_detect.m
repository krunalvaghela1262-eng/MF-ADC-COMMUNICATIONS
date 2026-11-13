function [a_hat, best_idx] = conventional_detect(x_q, HA, A)
%CONVENTIONAL_DETECT  Minimum-Euclidean ML detector for an unclipped conventional ADC.
%
%   a_hat = CONVENTIONAL_DETECT(x_q, HA, A) is the reference receiver: no
%   folding, a full-range quantizer (quantize_full_range) wide enough for
%   the whole signal swing, and a nearest-candidate search
%
%       a_hat = argmin_a || x_q - H a ||^2 .
%
%   It needs many more bits than the modulo-ADC to reach the same
%   resolution, and serves as the performance bound ("oracle").

    dist = sum((x_q - HA).^2, 1);
    [~, best_idx] = min(dist);
    a_hat = A(:, best_idx);
end
