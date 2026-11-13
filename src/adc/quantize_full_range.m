function [xq, step] = quantize_full_range(x, x_min, x_max, R_bit)
%QUANTIZE_FULL_RANGE  R-bit uniform quantizer spanning [x_min, x_max] (conventional ADC).
%
%   [xq, step] = QUANTIZE_FULL_RANGE(x, x_min, x_max, R_bit) is the
%   reference "conventional ADC": no folding, so its range must cover the
%   full signal swing, step = (x_max - x_min) / 2^R_bit. Values outside
%   the range clip to the outermost bin.
%
%   Used for the unclipped conventional-ADC oracle the detector is
%   compared against.

    L_levels = 2^R_bit;
    step     = (x_max - x_min) / L_levels;

    x       = min(max(x, x_min), x_max - eps(x_max));
    bin_idx = floor((x - x_min) / step);
    bin_idx = min(max(bin_idx, 0), L_levels - 1);
    xq      = x_min + (bin_idx + 0.5) * step;
end
