function S = injective_pam_alphabet(lambda)
%INJECTIVE_PAM_ALPHABET  8-ary PAM alphabets that stay distinguishable after folding.
%
%   S = INJECTIVE_PAM_ALPHABET(lambda) returns the constellation used in
%   the experiments for lambda = 1, 2 or 3. The modulo operator is
%   many-to-one, so the alphabet is chosen injective w.r.t. M_lambda:
%   no two symbols fold to the same residue (following Mulleti et al.,
%   ICASSP 2025). The peak amplitude c = max|S| gives the compression
%   ratio c/lambda ~= 10.4, 4.9, 3.3 respectively.

    switch lambda
        case 1
            S = [-8.875, -6.625, -4.375, -2.125, +2.125, +4.625, +6.875, +10.375];
        case 2
            S = [-9.75,  -5.25,  -4.75,  -0.25,  +0.25,  +4.75,  +5.25,  +9.75];
        case 3
            S = [-9.875, -7.875, -5.125, -0.375, +0.375, +5.125, +7.875, +9.875];
        otherwise
            error('injective_pam_alphabet: no stored alphabet for lambda = %g (use 1, 2 or 3).', lambda);
    end

    folded = sort(modulo_fold(S, lambda));
    assert(all(diff(folded) > 0), 'Alphabet is not injective under M_lambda.');
end
