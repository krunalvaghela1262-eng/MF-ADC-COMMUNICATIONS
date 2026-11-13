function res = mc_block_search_long_sequence(cfg)
%MC_BLOCK_SEARCH_LONG_SEQUENCE  Monte-Carlo SER for long sequences with Algorithm 1.
%
%   res = MC_BLOCK_SEARCH_LONG_SEQUENCE()      default: Ns = 64 symbols
%   res = MC_BLOCK_SEARCH_LONG_SEQUENCE(cfg)   override any field
%
%   At Ns = 64 the exhaustive search would need 8^64 ~ 6.3e57 hypotheses.
%   Algorithm 1 (mhbl_block_detect) searches blocks of Nb = 4 symbols
%   (8^4 = 4096 hypotheses each), cancelling interference from the rest
%   of the sequence with the previous pass' estimate, so the cost grows
%   linearly with Ns.
%
%   Receivers compared on the same analog signal per trial:
%     MHBL (block)          R_mhbl-bit modulo-ADC, Algorithm 1 (Jacobi)
%     Conventional (block)  R_conv-bit full-range ADC, block Euclidean search
%
%   Also records the mean decode time and number of passes per trial.

    if nargin < 1, cfg = struct(); end
    cfg = set_default(cfg, 'NoS',        64);
    cfg = set_default(cfg, 'W',          80);
    cfg = set_default(cfg, 'lambda',     1);
    cfg = set_default(cfg, 'sigma_lpf',  cfg.lambda/3);
    cfg = set_default(cfg, 'R_mhbl',     5);
    cfg = set_default(cfg, 'R_conv',     8);
    cfg = set_default(cfg, 'OF_vec',     [1 5 10 15 20 25]);
    cfg = set_default(cfg, 'K',          300);
    cfg = set_default(cfg, 'num_mc',     20);
    cfg = set_default(cfg, 'block_size', 4);    % Nb
    cfg = set_default(cfg, 'guard_sym',  1);    % edge symbols discarded per side
    cfg = set_default(cfg, 'margin_nyq', 3);    % eta, in Nyquist periods
    cfg = set_default(cfg, 'max_passes', 10);   % K in Algorithm 1
    cfg = set_default(cfg, 'mode',       'jacobi');
    cfg = set_default(cfg, 'seed',       1);
    cfg = set_default(cfg, 'plot',       true);

    rng(cfg.seed);
    S     = injective_pam_alphabet(cfg.lambda);
    c_max = max(ceil(abs(S)));
    A_blk = enumerate_sequences(S, cfg.block_size);
    op    = operating_point(S, cfg.lambda, cfg.sigma_lpf);

    [~, Delta] = quantize_modulo(0, cfg.lambda, cfg.R_mhbl);
    sigma_q    = Delta / sqrt(12);

    fprintf('[block search] Ns=%d, Nb=%d (%d hyp/block), lambda=%g, c/lambda=%.1f, SNR=%.1f dB, MC=%d\n', ...
        cfg.NoS, cfg.block_size, size(A_blk,2), cfg.lambda, op.c_over_lambda, op.snr_db, cfg.num_mc);
    fprintf('               exhaustive search would need |S|^Ns = %.2e hypotheses\n', numel(S)^cfg.NoS);

    nOF = numel(cfg.OF_vec);
    res.names  = {sprintf('MHBL block search (R=%d)', cfg.R_mhbl), ...
                  sprintf('Conventional ADC (R=%d)', cfg.R_conv)};
    res.OF     = cfg.OF_vec(:);
    res.ser    = zeros(nOF, 2);
    res.time_s = zeros(nOF, 2);
    res.passes = zeros(nOF, 2);
    res.cfg    = cfg;

    for oi = 1:nOF
        OF = cfg.OF_vec(oi);

        [~, t_d] = generate_pam_signal(S, cfg.NoS, cfg.W, cfg.K, zeros(cfg.NoS, 1));
        obs     = observation_grid(t_d, cfg.W, cfg.NoS, OF, cfg.K);
        H        = build_sinc_matrix(obs.t_s, cfg.W, cfg.NoS);
        Sigma_T  = noise_covariance(obs.t_s, cfg.W, cfg.lambda, cfg.sigma_lpf, sigma_q);
        blocks   = build_block_info(H, Sigma_T, obs.t_s, cfg.NoS, cfg.W, cfg);

        ser = zeros(cfg.num_mc, 2); tim = ser; pas = ser;
        for mc = 1:cfg.num_mc
            [f_cont, t_d, a_true] = generate_pam_signal(S, cfg.NoS, cfg.W, cfg.K);
            r = sample_signal(f_cont + bandlimited_noise(t_d, cfg.W, cfg.sigma_lpf), obs);

            % modulo-ADC + Algorithm 1
            y_lambda = quantize_modulo(modulo_fold(r, cfg.lambda), cfg.lambda, cfg.R_mhbl);
            a0       = peak_init(y_lambda, obs.t_s, cfg.W, cfg.NoS, S, cfg.lambda);
            t0 = tic;
            [a_mhbl, pas(mc,1)] = mhbl_block_detect(y_lambda, blocks, A_blk, cfg.lambda, ...
                                                   a0, cfg.max_passes, cfg.mode);
            tim(mc,1) = toc(t0);

            % conventional ADC + block search
            x_q = quantize_full_range(r, -c_max, c_max, cfg.R_conv);
            a0c = peak_init(x_q, obs.t_s, cfg.W, cfg.NoS, S);
            t0 = tic;
            [a_conv, pas(mc,2)] = conventional_block_detect(x_q, blocks, A_blk, a0c, cfg.max_passes);
            tim(mc,2) = toc(t0);

            ser(mc,:) = [symbol_error_rate(a_true, a_mhbl), symbol_error_rate(a_true, a_conv)];
        end

        res.ser(oi,:)    = mean(ser, 1);
        res.time_s(oi,:) = mean(tim, 1);
        res.passes(oi,:) = mean(pas, 1);
        fprintf('  OF=%2d (N=%4d, %d blocks): MHBL %.2f%% (%.0f ms, %.1f passes) | Conv %.2f%%\n', ...
            OF, obs.N, numel(blocks), res.ser(oi,1), 1e3*res.time_s(oi,1), res.passes(oi,1), res.ser(oi,2));
    end

    if cfg.plot
        figure('Color', 'w', 'Position', [100 100 800 520]); hold on; grid on; box on;
        plot(res.OF, res.ser(:,1), '-o', 'Color', [0.85 0.1 0.1], 'MarkerFaceColor', [0.85 0.1 0.1], ...
            'LineWidth', 2, 'DisplayName', res.names{1});
        plot(res.OF, res.ser(:,2), '-s', 'Color', [0 0 0], 'MarkerFaceColor', [0 0 0], ...
            'LineWidth', 2, 'DisplayName', res.names{2});
        xlabel('Oversampling factor (OF)'); ylabel('Mean SER (%)'); ylim([0 100]);
        xticks(res.OF); legend('Location', 'best');
        title(sprintf('Long sequence, N_s = %d, block search (N_b = %d)', cfg.NoS, cfg.block_size));
    end
end

function cfg = set_default(cfg, name, value)
    if ~isfield(cfg, name) || isempty(cfg.(name)), cfg.(name) = value; end
end
