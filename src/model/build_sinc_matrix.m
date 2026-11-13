function H = build_sinc_matrix(t_s, W, NoS)
%BUILD_SINC_MATRIX  Interpolation matrix H with [H]_{n,m} = sinc(2W (t_n - t_m)).
%
%   H = BUILD_SINC_MATRIX(t_s, W, NoS) maps a symbol vector a (NoS x 1) to
%   the noiseless oversampled samples f = H*a at the instants t_s.
%   At OF = 1 (samples exactly on the Nyquist instants) H = I.
%
%   Paper: Section 2, the model  y_lambda = M_lambda(H a + w) + q.

    t_m = symbol_times(NoS, W);
    H   = sinc(2*W*(t_s(:) - t_m.'));
end
