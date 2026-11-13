function res = folded_noise_autocorrelation(cfg)
%FOLDED_NOISE_AUTOCORRELATION  Theory vs simulation for the autocorrelation of folded noise.
%
%   res = FOLDED_NOISE_AUTOCORRELATION()      lambda/sigma in {4, 3, 2}
%   res = FOLDED_NOISE_AUTOCORRELATION(cfg)
%
%   Checks the Appendix-A result used to build Sigma_Y:
%
%       R_Y(tau) = E[ M_lambda(w(t)) M_lambda(w(t+tau)) ]
%                = sum_m A_{2m+1} rho(tau)^(2m+1),   rho(tau) = sinc(2W tau)
%
%   For each lambda/sigma_lpf it compares
%     - rho(tau)                      (unfolded noise)
%     - R_Y(tau)/R_Y(0) from the series (folded_noise_coefficients)
%     - R_Y(tau)/R_Y(0) estimated from simulated, folded bandlimited noise
%
%   For lambda/sigma >= 3 the folded and unfolded correlations coincide,
%   which is why Sigma_T ~= Sigma_w + sigma_q^2 I is accurate there.

    if nargin < 1, cfg = struct(); end
    cfg = set_default(cfg, 'ratios',   [4 3 2]);   % lambda / sigma_lpf
    cfg = set_default(cfg, 'W',        80);
    cfg = set_default(cfg, 'lambda',   1);
    cfg = set_default(cfg, 'K',        20);        % samples per Nyquist period
    cfg = set_default(cfg, 'duration', 200);       % seconds of simulated noise
    cfg = set_default(cfg, 'max_lag',  3);         % in Nyquist periods
    cfg = set_default(cfg, 'seed',     3);
    cfg = set_default(cfg, 'plot',     true);

    rng(cfg.seed);
    T       = 1/(2*cfg.W);
    Td      = T / cfg.K;
    t_d     = (0:Td:cfg.duration).';
    lag_max = cfg.max_lag * cfg.K;
    tau     = (-lag_max:lag_max).' * Td;
    rho     = sinc(2*cfg.W*tau);

    res.tau_over_T = tau / T;
    res.rho        = rho;
    res.ratios     = cfg.ratios;
    res.theory     = zeros(numel(tau), numel(cfg.ratios));
    res.empirical  = zeros(numel(tau), numel(cfg.ratios));

    for k = 1:numel(cfg.ratios)
        sigma = cfg.lambda / cfg.ratios(k);

        A_coeffs = folded_noise_coefficients(cfg.lambda, sigma);
        R_Y = zeros(size(tau));
        for m = 0:numel(A_coeffs)-1
            R_Y = R_Y + A_coeffs(m+1) * rho.^(2*m+1);
        end
        res.theory(:,k) = R_Y / R_Y(lag_max+1);

        w   = bandlimited_noise(t_d, cfg.W, sigma);
        y   = modulo_fold(w, cfg.lambda);
        r   = xcorr(y - mean(y), lag_max, 'unbiased');
        res.empirical(:,k) = r / r(lag_max+1);

        fprintf('lambda/sigma = %g : %d series terms, max |theory - sim| = %.3f\n', ...
            cfg.ratios(k), numel(A_coeffs), max(abs(res.theory(:,k) - res.empirical(:,k))));
    end

    if cfg.plot
        figure('Color', 'w', 'Position', [100 100 1200 360]);
        for k = 1:numel(cfg.ratios)
            subplot(1, numel(cfg.ratios), k); hold on; grid on; box on;
            plot(res.tau_over_T, rho, '--', 'Color', [0.1 0.35 0.75], 'LineWidth', 1.6);
            plot(res.tau_over_T, res.theory(:,k), '-', 'Color', [0.85 0.1 0.1], 'LineWidth', 1.6);
            plot(res.tau_over_T(1:4:end), res.empirical(1:4:end,k), 'k.', 'MarkerSize', 9);
            xlabel('\tau / T'); ylabel('normalized autocorrelation');
            title(sprintf('\\lambda / \\sigma_{lpf} = %g', cfg.ratios(k)));
            if k == 1, legend('\rho(\tau) unfolded', 'R_Y/R_Y(0) series', 'R_Y/R_Y(0) simulated'); end
        end
    end
end

function cfg = set_default(cfg, name, value)
    if ~isfield(cfg, name) || isempty(cfg.(name)), cfg.(name) = value; end
end
