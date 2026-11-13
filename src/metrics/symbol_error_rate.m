function ser = symbol_error_rate(a_true, a_hat)
%SYMBOL_ERROR_RATE  Percentage of wrongly detected symbols.
%
%   ser = SYMBOL_ERROR_RATE(a_true, a_hat) = 100/Ns * sum( a_hat_i ~= a_i ).
%   Averaging this over Monte-Carlo trials gives the "mean SER (%)"
%   reported in the paper (Section 4).

    a_true = a_true(:);
    a_hat  = a_hat(:);
    ser    = 100 * mean(a_true ~= a_hat);
end
