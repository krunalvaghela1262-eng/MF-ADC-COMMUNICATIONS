function run_smoke_tests()
%RUN_SMOKE_TESTS  Fast sanity checks for the whole pipeline (< 1 minute).
%   >> setup_paths; run_smoke_tests

    root = fileparts(fileparts(mfilename('fullpath')));
    addpath(genpath(fullfile(root, 'src')));
    addpath(fullfile(root, 'experiments'));
    rng(0);

    % 1. modulo operator: range and kernel
    x = 25 * randn(1000, 1); lambda = 1.3;
    y = modulo_fold(x, lambda);
    assert(all(y >= -lambda & y < lambda), 'fold range');
    assert(max(abs(modulo_fold(x + 2*lambda*randi([-5 5], 1000, 1), lambda) - y)) < 1e-9, 'fold kernel');
    fprintf('[ok] modulo_fold: range [-lambda, lambda) and 2*lambda*Z kernel\n');

    % 2. quantizer: error bounded by Delta/2
    [xq, D] = quantize_modulo(y, lambda, 3);
    assert(max(abs(xq - y)) <= D/2 + 1e-12, 'quantizer error');
    fprintf('[ok] quantize_modulo: |error| <= Delta/2\n');

    % 3. enumeration matches |S|^N and covers every sequence once
    A = enumerate_sequences([1 2 3], 3);
    assert(isequal(size(A), [3 27]) && size(unique(A.', 'rows'), 1) == 27, 'enumeration');
    fprintf('[ok] enumerate_sequences: 3^3 unique columns\n');

    % 4. H = I at OF = 1 on the Nyquist instants
    W = 80; NoS = 5; t_m = symbol_times(NoS, W);
    assert(norm(build_sinc_matrix(t_m, W, NoS) - eye(NoS)) < 1e-12, 'H at Nyquist');
    fprintf('[ok] build_sinc_matrix: H = I at the Nyquist instants\n');

    % 5. wrap cancellation: residual of the TRUE hypothesis equals the noise
    S = injective_pam_alphabet(1); K = 300;
    [f, t_d, a] = generate_pam_signal(S, 4, W, K);
    obs = observation_grid(t_d, W, 4, 10, K);
    H   = build_sinc_matrix(obs.t_s, W, 4);
    w   = 0.2 * randn(obs.N, 1);
    r   = modulo_fold(modulo_fold(sample_signal(f, obs) + w, 1) - H*a, 1);
    assert(max(abs(r - w)) < 1e-9, 'wrap cancellation');
    fprintf('[ok] wrap cancellation: M(y - Ha_true) = w exactly\n');

    % 6. noiseless, fine quantizer -> MHBL is error-free
    res = mc_ser_vs_of(struct('OF_vec', [5 10], 'num_mc', 5, 'sigma_lpf', 0, ...
                              'R_mhbl', 8, 'R_conv', 10, 'plot', false));
    assert(all(res.ser(:, 1) == 0), 'noiseless MHBL should be error-free');
    fprintf('[ok] mc_ser_vs_of: noiseless MHBL SER = 0\n');

    % 7. block search runs end-to-end on a short sequence
    res = mc_block_search_long_sequence(struct('NoS', 12, 'OF_vec', 10, 'num_mc', 2, 'plot', false));
    assert(all(isfinite(res.ser(:))), 'block search');
    fprintf('[ok] mc_block_search_long_sequence: runs end to end\n');

    % 8. demo
    demo_single_trial(struct('plot', false));
    fprintf('\nAll smoke tests passed.\n');
end
