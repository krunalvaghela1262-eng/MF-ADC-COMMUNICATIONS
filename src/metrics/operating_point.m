function op = operating_point(S, lambda, sigma_lpf)
%OPERATING_POINT  Compression ratio and pre-folding in-band SNR of a configuration.
%
%   op = OPERATING_POINT(S, lambda, sigma_lpf) returns
%       op.c_over_lambda  c / lambda with c = max|S|  (how hard the signal is folded)
%       op.snr_db         10 log10( mean(S.^2) / sigma_lpf^2 )
%       op.lambda_over_sigma
%
%   e.g. lambda = 1, sigma_lpf = lambda/3 gives (c/lambda, SNR) ~= (10.4, 25.7 dB).

    op.c_over_lambda     = max(abs(S)) / lambda;
    op.snr_db            = 10*log10(mean(S.^2) / sigma_lpf^2);
    op.lambda_over_sigma = lambda / sigma_lpf;
end
