function [xq, step] = quantize_modulo(x, lambda, R_bit)
%QUANTIZE_MODULO  R-bit mid-rise uniform quantizer over the folded range [-lambda, lambda).
%
%   [xq, step] = QUANTIZE_MODULO(x, lambda, R_bit) maps the folded samples
%   onto 2^R_bit levels with step  Delta = 2*lambda / 2^R_bit  and returns
%   the bin centres. The quantization error is ~ U(-Delta/2, Delta/2), so
%   its variance is  sigma_q^2 = Delta^2 / 12.
%
%   This is the low-dynamic-range ADC that sits after the modulo folder
%   (paper Fig. 1). Because the folded signal never leaves [-lambda, lambda),
%   only a few bits are needed regardless of how large the input is.

    L_levels = 2^R_bit;
    step     = 2*lambda / L_levels;

    x       = min(max(x, -lambda), lambda - eps(lambda));
    bin_idx = floor((x + lambda) / step);
    bin_idx = min(max(bin_idx, 0), L_levels - 1);
    xq      = -lambda + (bin_idx + 0.5) * step;
end
