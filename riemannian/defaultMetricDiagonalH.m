function G = defaultMetricDiagonalH(m, H_elbo, H_prior, J, sigma2, info) %#ok<INUSD>
% DEFAULTMETRICDIAGONALH
% Simple metric: diagonal of H_elbo + ridge.
%
% This is a cheap Riemannian-ish choice:
%   G = diag(max(diag(H_elbo), eps)) + λ I
% so the natural gradient dm_nat = G^{-1} g_elbo still
% rescales by local curvature but is more stable than full H^{-1}.

    d        = size(H_elbo,1);
    diagH    = diag(H_elbo);
    diagH    = max(diagH, 1e-8);
    lambda   = 1e-4;
    G        = diag(diagH) + lambda * eye(d);

end
