function x = sample_signal(x_cont, grid)
%SAMPLE_SIGNAL  Uniformly sample a dense-grid signal at the ADC instants.
%
%   x = SAMPLE_SIGNAL(x_cont, grid) returns the N samples of x_cont at the
%   instants grid.t_s produced by observation_grid.m (sampling rate
%   Fs = OF * 2W, restricted to the observation window).

    x = x_cont(grid.dense_idx);
end
