function out = demo_single_trial(cfg)
%DEMO_SINGLE_TRIAL  One transmission through the whole receiver, step by step.
%
%   out = DEMO_SINGLE_TRIAL()      runs with the defaults below
%   out = DEMO_SINGLE_TRIAL(cfg)   overrides any field of cfg
%
%   Pipeline (paper Fig. 1):
%     1. draw Ns symbols from an injective 8-PAM alphabet
%     2. build the sinc-pulse signal f(t) on a dense grid
%     3. add LPF-shaped channel noise w(t)
%     4. sample at OF x Nyquist inside the observation window
%     5. fold with M_lambda and quantize with an R-bit modulo-ADC
%     6. detect the symbols directly from the folded samples (MHBL)
%
%   Also plots the unfolded signal, the folded/quantized samples, and the
%   sent vs detected symbols.

    if nargin < 1, cfg = struct(); end
    cfg = set_default(cfg, 'NoS',       4);
    cfg = set_default(cfg, 'W',         80);       % one-sided bandwidth (Hz)
    cfg = set_default(cfg, 'lambda',    1);        % folding threshold
    cfg = set_default(cfg, 'OF',        10);       % oversampling factor
    cfg = set_default(cfg, 'K',         300);      % dense-grid factor (OF must divide K)
    cfg = set_default(cfg, 'R_bit',     3);        % modulo-ADC resolution
    cfg = set_default(cfg, 'sigma_lpf', cfg.lambda/3);
    cfg = set_default(cfg, 'seed',      7);
    cfg = set_default(cfg, 'plot',      true);

    rng(cfg.seed);
    S  = injective_pam_alphabet(cfg.lambda);
    op = operating_point(S, cfg.lambda, cfg.sigma_lpf);
    fprintf('c/lambda = %.1f, in-band SNR = %.1f dB, lambda/sigma = %.1f\n', ...
        op.c_over_lambda, op.snr_db, op.lambda_over_sigma);

    % --- 1-3. transmit + channel ---
    [f_cont, t_d, a_true] = generate_pam_signal(S, cfg.NoS, cfg.W, cfg.K);
    r_cont = f_cont + bandlimited_noise(t_d, cfg.W, cfg.sigma_lpf);

    % --- 4. sample ---
    obs = observation_grid(t_d, cfg.W, cfg.NoS, cfg.OF, cfg.K);
    r    = sample_signal(r_cont, obs);

    % --- 5. modulo-ADC ---
    [y_lambda, Delta] = quantize_modulo(modulo_fold(r, cfg.lambda), cfg.lambda, cfg.R_bit);
    sigma_q = Delta / sqrt(12);

    % --- 6. MHBL detection ---
    A       = enumerate_sequences(S, cfg.NoS);
    H       = build_sinc_matrix(obs.t_s, cfg.W, cfg.NoS);
    Sigma_T = noise_covariance(obs.t_s, cfg.W, cfg.lambda, cfg.sigma_lpf, sigma_q);
    a_hat   = mhbl_detect(y_lambda, H*A, A, safe_cholesky(Sigma_T), cfg.lambda);

    fprintf('N = %d samples, %d candidate sequences, %d-bit ADC on [-%g, %g)\n', ...
        obs.N, size(A,2), cfg.R_bit, cfg.lambda, cfg.lambda);
    fprintf('sent     : %s\n', mat2str(a_true.', 4));
    fprintf('detected : %s\n', mat2str(a_hat.', 4));
    fprintf('SER      : %.1f %%\n', symbol_error_rate(a_true, a_hat));

    out = struct('a_true', a_true, 'a_hat', a_hat, 't_s', obs.t_s, ...
                 'r', r, 'y_lambda', y_lambda, 'cfg', cfg);

    if cfg.plot
        win = t_d > obs.t_s(1) - 2e-3 & t_d < obs.t_s(end) + 2e-3;
        t_m = symbol_times(cfg.NoS, cfg.W);
        figure('Color', 'w', 'Position', [100 100 900 650]);

        subplot(3,1,1); hold on; grid on; box on;
        plot(t_d(win), r_cont(win), 'Color', [0.2 0.4 0.8], 'LineWidth', 1.5);
        stem(obs.t_s, r, 'filled', 'MarkerSize', 3, 'Color', [0.2 0.4 0.8]);
        yline([-cfg.lambda cfg.lambda], 'r--', 'LineWidth', 1.2);
        title('Received signal r(t) = f(t) + w(t)  (high dynamic range)');

        subplot(3,1,2); hold on; grid on; box on;
        stairs(obs.t_s, y_lambda, 'Color', [0.85 0.1 0.1], 'LineWidth', 1.2);
        plot(obs.t_s, y_lambda, '.', 'Color', [0.85 0.1 0.1], 'MarkerSize', 10);
        yline([-cfg.lambda cfg.lambda], 'r--', 'LineWidth', 1.2);
        ylim(1.3*[-cfg.lambda cfg.lambda]);
        title(sprintf('Folded + %d-bit quantized samples y_\\lambda  (all inside [-\\lambda, \\lambda))', cfg.R_bit));

        subplot(3,1,3); hold on; grid on; box on;
        stem(t_m, a_true, 'filled', 'LineWidth', 1.5, 'Color', [0 0 0]);
        stem(t_m, a_hat, 'LineWidth', 1.5, 'Color', [0.1 0.65 0.3], 'LineStyle', '--');
        legend('sent', 'detected by MHBL', 'Location', 'best');
        xlabel('time (s)');
        title('Symbols recovered directly from folded samples -- no unfolding');
    end
end

function cfg = set_default(cfg, name, value)
    if ~isfield(cfg, name) || isempty(cfg.(name)), cfg.(name) = value; end
end
