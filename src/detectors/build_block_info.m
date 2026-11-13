function blocks = build_block_info(H, Sigma_T, t_s, NoS, W, cfg)
%BUILD_BLOCK_INFO  Precompute the overlapping blocks used by Algorithm 1.
%
%   blocks = BUILD_BLOCK_INFO(H, Sigma_T, t_s, NoS, W, cfg) partitions the
%   NoS symbols into overlapping blocks of Nb = cfg.block_size symbols.
%   Each block j stores everything that does not change between trials
%   (paper Algorithm 1, "precomputed block parameters"):
%
%       .sym      C_j     symbol indices searched jointly in block j
%       .win      I_j     sample indices: block span +/- margin
%       .Hin      H(I_j, C_j)
%       .Hout     H(I_j, :) restricted to symbols outside the block
%       .out_idx  indices of those outside symbols
%       .L        Cholesky factor of Sigma_T(I_j, I_j)  (local precision)
%       .commit   positions inside the block whose estimates are kept
%
%   cfg fields
%       block_size   Nb, symbols per block           (default 4)
%       guard_sym    edge symbols discarded per side (default 1)
%       margin_nyq   sample margin eta, in Nyquist periods (default 3)
%
%   Blocks advance by the core length Nb - 2*guard_sym so committed cores
%   tile the sequence exactly once; the first/last block also commit
%   their outer edges so no symbol is left without an owner.

    if ~isfield(cfg, 'block_size'), cfg.block_size = 4; end
    if ~isfield(cfg, 'guard_sym'),  cfg.guard_sym  = 1; end
    if ~isfield(cfg, 'margin_nyq'), cfg.margin_nyq = 3; end

    Nb     = cfg.block_size;
    G      = cfg.guard_sym;
    T      = 1 / (2*W);
    margin = cfg.margin_nyq * T;
    t_m    = symbol_times(NoS, W);

    assert(Nb <= NoS, 'block_size must not exceed the number of symbols.');
    stride = max(Nb - 2*G, 1);
    last   = NoS - Nb + 1;
    starts = unique([1:stride:last, last]);

    blocks = struct('start', {}, 'sym', {}, 'win', {}, 'Hin', {}, 'Hout', {}, ...
                    'out_idx', {}, 'L', {}, 'commit', {});

    for b = 1:numel(starts)
        s    = starts(b);
        sym  = s:(s + Nb - 1);
        oidx = setdiff(1:NoS, sym);

        win = find(t_s >= t_m(sym(1)) - margin & t_s <= t_m(sym(end)) + margin);
        if numel(win) < Nb, win = (1:numel(t_s)).'; end

        commit = (1 + G):(Nb - G);
        if s == 1,    commit = [1:G, commit];          end
        if s == last, commit = [commit, (Nb-G+1):Nb];  end

        blocks(b).start   = s;
        blocks(b).sym     = sym;
        blocks(b).win     = win;
        blocks(b).Hin     = H(win, sym);
        blocks(b).Hout    = H(win, oidx);
        blocks(b).out_idx = oidx;
        blocks(b).L       = safe_cholesky(Sigma_T(win, win));
        blocks(b).commit  = unique(commit);
    end
end
