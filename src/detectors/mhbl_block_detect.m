function [a_hat, passes, status] = mhbl_block_detect(y_lambda, blocks, A_blk, lambda, a_init, max_passes, mode)
%MHBL_BLOCK_DETECT  Block-decomposed Mahalanobis search (paper Algorithm 1).
%
%   [a_hat, passes, status] = MHBL_BLOCK_DETECT(y_lambda, blocks, A_blk,
%       lambda, a_init, max_passes, mode)
%
%   Exhaustive MHBL costs |S|^Ns hypotheses (8^64 ~ 6e57 for Ns = 64).
%   Instead, each block j searches only its Nb local symbols, with the
%   interference from all other symbols cancelled using the previous
%   estimate:
%
%       d_j    = H(I_j, outside) * a_hat(outside)           (eq. 12)
%       y_eff  = y_lambda(I_j) - d_j
%       r_j(a) = M_lambda( y_eff - H(I_j, C_j) a )
%       a_j    = argmin_{a in S^Nb}  || L_j \ r_j(a) ||^2    (eq. 13)
%
%   Edge symbols of each block are discarded and only the core is
%   committed; passes repeat until the estimate stops changing.
%   Cost grows LINEARLY in Ns:  O(K * P * OF^2 (Nb + 2 eta)^2 |S|^Nb).
%
%   Inputs
%       y_lambda    folded + quantized samples (whole sequence)
%       blocks      from build_block_info
%       A_blk       enumerate_sequences(S, Nb)
%       a_init      initial estimate (e.g. peak_init)
%       max_passes  K, maximum number of passes (default 10)
%       mode        'jacobi' (default) -- every block uses the estimate
%                   from the previous pass, so all blocks can run in
%                   parallel (Algorithm 1 as written);
%                   'gs' -- Gauss-Seidel, update in place (sequential).
%
%   status: 1 = converged (fixed point), 2 = entered a limit cycle,
%           3 = hit max_passes.
%
%   Note: folding is applied to the FULL hypothesis (block + context),
%   because M_lambda is not additive across individual symbols.

    if nargin < 6 || isempty(max_passes), max_passes = 10; end
    if nargin < 7 || isempty(mode),       mode = 'jacobi'; end
    use_jacobi = strcmpi(mode, 'jacobi');

    a_hat   = a_init(:);
    NoS     = numel(a_hat);
    history = zeros(NoS, max_passes);
    status  = 3;
    passes  = 0;

    for p = 1:max_passes
        a_prev = a_hat;
        a_new  = a_hat;

        for b = 1:numel(blocks)          % independent blocks -> parfor-able in Jacobi mode
            B = blocks(b);
            if use_jacobi, ctx = a_prev(B.out_idx); else, ctx = a_new(B.out_idx); end

            y_eff = y_lambda(B.win) - B.Hout * ctx;
            R     = modulo_fold(y_eff - B.Hin * A_blk, lambda);
            d     = sum((B.L \ R).^2, 1);
            [~, k] = min(d);

            a_new(B.start + B.commit - 1) = A_blk(B.commit, k);
        end

        a_hat  = a_new;
        passes = p;

        history(:, p) = a_hat;
        if p > 1
            match = find(all(history(:, 1:p-1) == a_hat, 1), 1, 'last');
            if ~isempty(match)
                status = 1 + (match ~= p-1);
                break;
            end
        end
    end
end
