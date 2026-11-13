function A_coeffs = folded_noise_coefficients(lambda, sigma_lpf)
%FOLDED_NOISE_COEFFICIENTS  Series coefficients of the folded-noise autocorrelation.
%
%   A_coeffs = FOLDED_NOISE_COEFFICIENTS(lambda, sigma_lpf) returns
%   [A_1; A_3; A_5; ...] such that, for w(t) Gaussian with correlation
%   rho(tau) = sinc(2W tau),
%
%       R_Y(tau) = E[ M_lambda(w(t)) M_lambda(w(t+tau)) ]
%                = sum_m  A_{2m+1} * rho(tau)^(2m+1)
%
%   with  alpha = sigma_lpf^2 pi^2 / (2 lambda^2),
%         A_{2m+1} = lambda^2 (2 alpha)^(2m+1) / (pi^2 (2m+1)!) * B_m^2,
%         B_m      = 2 * sum_{n>=1} (-1)^n n^(2m) exp(-alpha n^2).
%
%   Derivation: Fourier series of the sawtooth M_lambda + the bivariate
%   Gaussian characteristic function (paper Appendix A, eq. 16-23).
%
%   The alternating sum B_m suffers catastrophic cancellation in double
%   precision, so it is evaluated with 50-digit vpa when the Symbolic Math
%   Toolbox is available. The series is truncated once a term falls below
%   1e-14 of the leading term.

    max_n     = 150;
    max_m_cap = 40;
    rel_tol   = 1e-14;

    if sigma_lpf == 0
        A_coeffs = 0;          % no channel noise -> no folded-noise term
        return;
    end

    alpha    = (sigma_lpf^2 * pi^2) / (2 * lambda^2);
    A_coeffs = zeros(max_m_cap + 1, 1);
    use_vpa  = exist('vpa', 'file') == 2;

    if use_vpa
        digits_old = digits(50);
        cleanup    = onCleanup(@() digits(digits_old));
        alpha_v    = vpa(alpha);
        lambda_v   = vpa(lambda);
        n_v        = vpa(1:max_n).';
    else
        warning('folded_noise_coefficients:noVPA', ...
            'Symbolic Math Toolbox not found -- using double precision (less accurate for small lambda/sigma).');
        n_d = (1:max_n).';
    end

    for m = 0:max_m_cap
        k = 2*m + 1;
        if use_vpa
            B     = 2 * sum(((-1).^n_v) .* (n_v.^(2*m)) .* exp(-alpha_v .* n_v.^2));
            coeff = lambda_v^2 * (2*alpha_v)^k / (pi^2 * factorial(sym(k))) * B^2;
            A_coeffs(m+1) = double(abs(coeff));
        else
            B     = 2 * sum(((-1).^n_d) .* (n_d.^(2*m)) .* exp(-alpha .* n_d.^2));
            A_coeffs(m+1) = abs(lambda^2 * (2*alpha)^k / (pi^2 * factorial(k)) * B^2);
        end

        if m > 2 && A_coeffs(m+1) < rel_tol * A_coeffs(1)
            A_coeffs = A_coeffs(1:m+1);
            break;
        end
    end
end
