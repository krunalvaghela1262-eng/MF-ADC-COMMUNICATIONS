function a0 = peak_init(x, t_s, W, NoS, S, lambda)
%PEAK_INIT  Context-free initial estimate from the samples nearest each Nyquist instant.
%
%   a0 = PEAK_INIT(x, t_s, W, NoS, S, lambda) slices the sample closest to
%   each symbol centre t_m. Since sinc(2W(t - t_k)) vanishes at t_m for
%   k ~= m, that sample carries symbol m with (almost) no ISI.
%
%   With lambda given, x is treated as folded and compared against the
%   folded alphabet M_lambda(S) using a wrapped distance (MHBL init).
%   With lambda omitted or [], x is unfolded (conventional-ADC init).
%
%   A good start matters for MHBL: a context error larger than lambda
%   folds to an arbitrary residue, so starting from zeros would make the
%   first pass pure noise.

    t_m    = symbol_times(NoS, W);
    pk_idx = zeros(NoS, 1);
    for m = 1:NoS
        [~, pk_idx(m)] = min(abs(t_s - t_m(m)));
    end

    S = S(:).';
    if nargin >= 6 && ~isempty(lambda)
        R = modulo_fold(x(pk_idx) - modulo_fold(S, lambda), lambda);
    else
        R = x(pk_idx) - S;
    end
    [~, k] = min(abs(R), [], 2);
    a0 = S(k).';
end
