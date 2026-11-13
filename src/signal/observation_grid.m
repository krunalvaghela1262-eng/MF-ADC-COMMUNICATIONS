function grid = observation_grid(t_d, W, NoS, OF, K)
%OBSERVATION_GRID  Sampling instants of the oversampled ADC inside the window.
%
%   grid = OBSERVATION_GRID(t_d, W, NoS, OF, K) decimates the dense grid
%   t_d by K/OF (sampling rate Fs = OF * 2W) and keeps only the samples
%   that fall strictly inside the observation window
%
%       ( t_1 - T ,  t_NoS + T )
%
%   i.e. between the main-lobe edges of the first and last sinc pulse
%   (paper Fig. 2). Returned fields:
%
%       grid.t_s        N x 1 sampling instants
%       grid.dense_idx  N x 1 indices into t_d, so that  x = x_cont(grid.dense_idx)
%       grid.N          number of samples in the window
%       grid.Ts         sampling period
%
%   Sampling a dense-grid signal is then just  sample_signal(x_cont, grid).

    alpha = K / OF;
    assert(abs(alpha - round(alpha)) < 1e-12, ...
        'observation_grid: OF = %g must divide the dense-grid factor K = %g.', OF, K);
    alpha = round(alpha);

    T   = 1 / (2*W);
    t_m = symbol_times(NoS, W);
    lo  = t_m(1)   - T;
    hi  = t_m(end) + T;

    all_idx = (1 : alpha : numel(t_d)).';
    t_all   = t_d(all_idx);
    keep    = (t_all > lo + 1e-12) & (t_all < hi - 1e-12);

    grid.dense_idx = all_idx(keep);
    grid.t_s       = t_all(keep);
    grid.N         = numel(grid.t_s);
    grid.Ts        = T / OF;
end
