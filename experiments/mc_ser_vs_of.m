function res = mc_ser_vs_of(cfg)
%MC_SER_VS_OF  Monte-Carlo SER vs oversampling factor (exhaustive search, short sequences).
%
%   res = MC_SER_VS_OF()      runs the default configuration
%   res = MC_SER_VS_OF(cfg)   overrides any field (see defaults below)
%
%   For every OF in cfg.OF_vec and every Monte-Carlo trial, one noisy
%   transmission is generated and detected by three receivers that all
%   see the SAME analog signal:
%
%     MHBL          fold -> R_mhbl-bit modulo-ADC -> Mahalanobis ML (eq. 11)
%     MHBL, Sigma=I same folded samples, Euclidean metric on the wrapped
%                   residual (ablation: ignores the noise correlation)
%     Conventional  no fold -> R_conv-bit full-range ADC -> Euclidean ML
%
%   SER is averaged over cfg.num_mc trials per OF. The deterministic
%   pieces (sample grid, H, candidate set, Sigma_T and its Cholesky
%   factor) are computed once per OF and reused by every trial.
%   Trials run in a parfor loop (serial if no parallel pool is open).
%
%   Example:  res = mc_ser_vs_of(struct('lambda', 2, 'num_mc', 200));

    if nargin < 1, cfg = struct(); end
    cfg = set_default(cfg, 'NoS',       4);
    cfg = set_default(cfg, 'W',         80);
    cfg = set_default(cfg, 'lambda',    1);
    cfg = set_default(cfg, 'sigma_lpf', cfg.lambda/3);
    cfg = set_default(cfg, 'R_mhbl',    3);
    cfg = set_default(cfg, 'R_conv',    6);
    cfg = set_default(cfg, 'OF_vec',    [2 5 10 15 20 25 30]);
    cfg = set_default(cfg, 'K',         300);
    cfg = set_default(cfg, 'num_mc',    100);
    cfg = set_default(cfg, 'seed',      1);
    cfg = set_default(cfg, 'plot',      true);

    rng(cfg.seed);
    S     = injective_pam_alphabet(cfg.lambda);
    c_max = max(ceil(abs(S)));
    A     = enumerate_sequences(S, cfg.NoS);
    op    = operating_point(S, cfg.lambda, cfg.sigma_lpf);

    [~, Delta] = quantize_modulo(0, cfg.lambda, cfg.R_mhbl);
    sigma_q    = Delta / sqrt(12);

    fprintf('[mc_ser_vs_of] lambda=%g  c/lambda=%.1f  SNR=%.1f dB  Ns=%d  |S|^Ns=%d  MC=%d\n', ...
        cfg.lambda, op.c_over_lambda, op.snr_db, cfg.NoS, size(A,2), cfg.num_mc);

    nOF = numel(cfg.OF_vec);
    res.names = {sprintf('MHBL (R=%d)', cfg.R_mhbl), ...
                 sprintf('MHBL, \\Sigma=I (R=%d)', cfg.R_mhbl), ...
                 sprintf('Conventional ADC (R=%d)', cfg.R_conv)};
    res.ser   = zeros(nOF, 3);
    res.OF    = cfg.OF_vec(:);
    res.cfg   = cfg;
    res.op    = op;

    for oi = 1:nOF
        OF   = cfg.OF_vec(oi);
        tic;

        % --- per-OF precomputation (no randomness) ---
        [~, t_d] = generate_pam_signal(S, cfg.NoS, cfg.W, cfg.K, zeros(cfg.NoS, 1));
        obs     = observation_grid(t_d, cfg.W, cfg.NoS, OF, cfg.K);
        HA       = build_sinc_matrix(obs.t_s, cfg.W, cfg.NoS) * A;
        L_T      = safe_cholesky(noise_covariance(obs.t_s, cfg.W, cfg.lambda, cfg.sigma_lpf, sigma_q));
        L_I      = eye(obs.N);

        ser_runs = zeros(cfg.num_mc, 3);
        NoS = cfg.NoS; W = cfg.W; K = cfg.K; lambda = cfg.lambda;
        sig = cfg.sigma_lpf; Rm = cfg.R_mhbl; Rc = cfg.R_conv;

        parfor mc = 1:cfg.num_mc
            % transmit + channel + sample
            [f_cont, t_dd, a_true] = generate_pam_signal(S, NoS, W, K);
            r = sample_signal(f_cont + bandlimited_noise(t_dd, W, sig), obs);

            % modulo-ADC receivers
            y_lambda = quantize_modulo(modulo_fold(r, lambda), lambda, Rm);
            a_mhbl   = mhbl_detect(y_lambda, HA, A, L_T, lambda);
            a_eucl   = mhbl_detect(y_lambda, HA, A, L_I, lambda);

            % conventional (unclipped) ADC receiver
            x_q    = quantize_full_range(r, -c_max, c_max, Rc);
            a_conv = conventional_detect(x_q, HA, A);

            ser_runs(mc, :) = [symbol_error_rate(a_true, a_mhbl), ...
                               symbol_error_rate(a_true, a_eucl), ...
                               symbol_error_rate(a_true, a_conv)];
        end

        res.ser(oi, :) = mean(ser_runs, 1);
        fprintf('  OF=%2d (N=%3d): MHBL %6.2f%% | MHBL Sigma=I %6.2f%% | Conv %6.2f%%  [%.1fs]\n', ...
            OF, obs.N, res.ser(oi,1), res.ser(oi,2), res.ser(oi,3), toc);
    end

    if cfg.plot
        plot_ser(res, sprintf('SER vs OF  (N_s=%d, \\lambda=%g, c/\\lambda=%.1f, SNR=%.1f dB, %d trials)', ...
            cfg.NoS, cfg.lambda, op.c_over_lambda, op.snr_db, cfg.num_mc));
    end
end

function plot_ser(res, ttl)
    colors  = [0.85 0.10 0.10; 0.95 0.55 0.10; 0 0 0];
    styles  = {'-o', '--s', '-^'};
    figure('Color', 'w', 'Position', [100 100 800 520]); hold on; grid on; box on;
    for k = 1:size(res.ser, 2)
        plot(res.OF, res.ser(:,k), styles{k}, 'Color', colors(k,:), ...
            'MarkerFaceColor', colors(k,:), 'LineWidth', 2, 'MarkerSize', 7, ...
            'DisplayName', res.names{k});
    end
    xlabel('Oversampling factor (OF)'); ylabel('Mean SER (%)');
    ylim([0 100]); xticks(res.OF); legend('Location', 'best'); title(ttl);
end

function cfg = set_default(cfg, name, value)
    if ~isfield(cfg, name) || isempty(cfg.(name)), cfg.(name) = value; end
end
