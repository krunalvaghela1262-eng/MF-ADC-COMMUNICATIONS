function t_m = symbol_times(NoS, W)
%SYMBOL_TIMES  Centres of the NoS sinc pulses, spaced one Nyquist period apart.
%
%   t_m = SYMBOL_TIMES(NoS, W) returns t_m = (m - floor((NoS+1)/2)) * T for
%   m = 1..NoS, with T = 1/(2W). The block of pulses is centred on t = 0.
%   Since sinc(2W(t - t_m)) is zero at every other t_k, the noiseless
%   signal at t_m equals a_m exactly (no ISI at the Nyquist instants).

    T   = 1 / (2*W);
    t_m = ((1:NoS).' - floor((NoS+1)/2)) * T;
end
