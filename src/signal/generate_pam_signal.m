function [x_cont, t_d, symbols] = generate_pam_signal(S, NoS, W, K, symbols)
%GENERATE_PAM_SIGNAL  Sinc-pulse PAM transmit signal on a dense time grid.
%
%   [x_cont, t_d, symbols] = GENERATE_PAM_SIGNAL(S, NoS, W, K)
%   draws NoS symbols uniformly from the alphabet S and builds
%
%       f(t) = sum_m  a_m * sinc(2W (t - t_m)),     t_m = symbol_times(NoS, W)
%
%   on a dense grid t_d with spacing Td = T/K, where T = 1/(2W) is the
%   Nyquist (symbol) period. Because the dense grid is K times finer than
%   Nyquist, any oversampling factor OF that divides K is obtained exactly
%   by decimation (see observation_grid.m).
%
%   Pass a fifth argument to transmit a fixed symbol vector instead of a
%   random draw.
%
%   Paper: Section 2, f(t) = sum_m a_m g(t - mT) with g(t) = sinc(2Wt).

    if nargin < 5 || isempty(symbols)
        symbols = S(randi(numel(S), NoS, 1));
    end
    symbols = symbols(:);

    T   = 1 / (2*W);
    Td  = T / K;
    t_d = (-1 : Td : 1).';

    t_m    = symbol_times(NoS, W);
    x_cont = zeros(numel(t_d), 1);
    for m = 1:NoS
        x_cont = x_cont + symbols(m) * sinc(2*W*(t_d - t_m(m)));
    end
end
