function A = enumerate_sequences(S, N)
%ENUMERATE_SEQUENCES  All |S|^N candidate symbol vectors, one per column.
%
%   A = ENUMERATE_SEQUENCES(S, N) returns an N x |S|^N matrix whose
%   columns are every sequence in S^N (first row varies fastest). This is
%   the hypothesis set searched by the exhaustive ML detector, and by each
%   local block search (N = Nb) in Algorithm 1.
%
%   Toolbox-free replacement for combvec.

    M   = numel(S);
    idx = cell(1, N);
    [idx{:}] = ndgrid(1:M);

    A = zeros(N, M^N);
    for k = 1:N
        A(k, :) = S(idx{k}(:));
    end
end
