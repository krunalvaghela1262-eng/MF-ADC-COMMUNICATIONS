function [a_hat, passes] = conventional_block_detect(x_q, blocks, A_blk, a_init, max_passes)
%CONVENTIONAL_BLOCK_DETECT  Block-wise minimum-Euclidean search for the conventional ADC.
%
%   Same block structure as mhbl_block_detect, but on unfolded,
%   full-range-quantized samples and with a plain Euclidean metric.
%   Used as the reference receiver for long sequences.

    if nargin < 5 || isempty(max_passes), max_passes = 10; end

    a_hat  = a_init(:);
    passes = 0;
    for p = 1:max_passes
        a_prev = a_hat;
        for b = 1:numel(blocks)
            B = blocks(b);
            R = x_q(B.win) - (B.Hin * A_blk + B.Hout * a_hat(B.out_idx));
            [~, k] = min(sum(R.^2, 1));
            a_hat(B.start + B.commit - 1) = A_blk(B.commit, k);
        end
        passes = p;
        if isequal(a_hat, a_prev), break; end
    end
end
