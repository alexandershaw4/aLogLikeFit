function C = propose_deltaF_ranked(I, P, opts)
% I      : current active set (indices)
% P      : struct with fields J (N×p), r (N×1), S0 (p×p or diag), Lambda (p×1 ARD)
%          optional: W (N×N) residual precision; else assume I
% opts   : has max_active, cooldown, etc.
%
% Returns a column vector of candidate indices (not in I), sorted by ΔF_add desc.

p = size(P.J,2);
all = (1:p)';
J  = P.J; r = P.r;
if isfield(P,'W'), W = P.W; else, W = []; end

notI = setdiff(all, I);
% skip cooling params
if isfield(P,'cooldown') && ~isempty(P.cooldown)
    notI = setdiff(notI, find(P.cooldown>0));
end

if isempty(notI), C = notI; return; end

% Per-parameter projected utility: u_j ≈ 0.5 * (g_j^2 / H_jj)
% g_j = J_j' * r  (or J_j' W r)
% H_jj ≈ J_j' J_j + (S0^-1 + Lambda)_jj  (Gauss–Newton + prior/ARD)
if isempty(W)
    g = J' * r;
    Hdiag = sum(J.^2,1)';  % J'J diagonal
else
    Jr = J' * (W * r);
    g = Jr;
    Hdiag = sum(J .* (W*J), 1)'; % diag(J' W J)
end

% Add prior/ARD curvature
if isvector(P.S0)
    prior_prec = 1./max(P.S0, eps);
elseif ~isempty(P.S0) && isdiag(P.S0)
    prior_prec = 1./max(diag(P.S0), eps);
elseif ~isempty(P.S0)
    prior_prec = 1./max(diag(P.S0), 1e-12);
else
    prior_prec = zeros(p,1);
end
if isfield(P,'Lambda') && ~isempty(P.Lambda)
    prior_prec = prior_prec + P.Lambda(:);
end

Hjj = Hdiag + prior_prec;
u   = 0.5 * (g.^2 ./ max(Hjj, 1e-12));

% Optional: z-score screen before ΔF ranking
if isfield(opts,'use_BH') && opts.use_BH
    z = abs(g) ./ sqrt(max(Hjj,1e-12));
    keep = bh_mask(z(notI), opts.q_fdr);
    notI = notI(keep);
end

[~,ord] = sort(u(notI), 'descend');
C = notI(ord);
end

function m = bh_mask(z, q)
% two-sided p from z, BH at level q
p = 2*erfc(abs(z)/sqrt(2)) / 2;  % 2*(1 - Phi(|z|))
[ps,idx] = sort(p(:));
mtmp = false(size(p));
k = find(ps <= ( (1:numel(ps))' / numel(ps) ) * q, 1, 'last');
if ~isempty(k), mtmp(idx(1:k)) = true; end
m = mtmp;
end
