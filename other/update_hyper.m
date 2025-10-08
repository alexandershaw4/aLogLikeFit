function [w, Cw, h, sigma2, L3, stats] = update_hyper(e, Phi, w, iwC, w0, opts)
%UPDATE_HYPER  Newton-ascent update for heteroscedastic log-variance hyperparameters.
%
%  h = Phi * w,   sigma2 = exp(h),   iS = diag(exp(-h))
%
%  Maximises:  L_h = 0.5*( log|iS| - e' iS e )  - 0.5*(w-w0)' iwC (w-w0)
%  Using a GN/Newton step in w-space with optional backtracking.
%
% Inputs
%   e    : residuals (n x 1)
%   Phi  : basis/design for log-variance (n x q)
%   w    : current hyperparameter weights (q x 1)
%   iwC  : prior precision over w (q x q) [symmetric PD]
%   w0   : prior mean over w (q x 1)
%   opts : struct with optional fields:
%          .maxIter     (default 1) inner Newton steps
%          .backtrack   (default true)
%          .maxTries    (default 8)
%          .stepScale   (default 0.5) backtracking factor
%          .verbose     (default false)
%
% Outputs
%   w      : updated weights
%   Cw     : approx posterior covariance over w  (Cw ≈ A^{-1})
%   h      : updated log-variance (n x 1)
%   sigma2 : updated variances   (n x 1)
%   L3     : SPM-style reporting term:
%            0.5*log|iwC*Cw| - 0.5*(w-w0)' iwC (w-w0)
%   stats  : struct with fields {A, b, g_total, L_before, L_after, tries}
%
% Notes
%   - This routine treats heteroscedasticity via a low-dim basis Phi.
%   - If you want per-sample h (no basis), set Phi = eye(n), iwC = lambda*I.
%   - Plug this into your outer loop after updating m, using current residuals e.
%
% AS 2025

if nargin < 6, opts = struct; end
if ~isfield(opts,'maxIter'),    opts.maxIter  = 1;    end
if ~isfield(opts,'backtrack'),  opts.backtrack= true; end
if ~isfield(opts,'maxTries'),   opts.maxTries = 8;    end
if ~isfield(opts,'stepScale'),  opts.stepScale= 0.5;  end
if ~isfield(opts,'verbose'),    opts.verbose  = false;end

n = numel(e);
q = numel(w);
one_n = ones(n,1);

% helper to evaluate L_h quickly
    function [Lh, e2exp] = L_obj(w_eval)
        h_eval     = Phi * w_eval;
        invS_diag  = exp(-h_eval);          % diag(iS)
        logdet_iS  = -sum(h_eval);          % log|iS|
        quad_e     = sum((e.^2) .* invS_diag);
        L1         = 0.5*(logdet_iS - quad_e);
        Lprior     = -0.5*(w_eval - w0)' * (iwC * (w_eval - w0));
        Lh         = L1 + Lprior;
        if nargout>1
            e2exp = (e.^2) .* exp(-h_eval);
        end
    end

% initial objective
[L_before, e2exp] = L_obj(w);

tries_total = 0;

for it = 1:opts.maxIter
    % --- Gradient and (negative) Hessian in w-space ---
    % In h-space:
    %   g1_h = 0.5*(-1 + e.^2 .* exp(-h))
    %  -H1_h = 0.5*diag(e.^2 .* exp(-h))   [so H1_h is negative definite]
    %
    % Map to w: g1_w = Phi' * g1_h
    %           -H1_w = 0.5 * Phi' * diag(e2exp) * Phi
    % Prior in w: gprior_w = - iwC*(w - w0)
    %             -Hprior_w = iwC
    %
    % Newton-ascent step solves:  A * dw = b
    %   A = (-H_total) = 0.5*Phi'*diag(e2exp)*Phi + iwC
    %   b = (-g_total) = 0.5*Phi'*(one_n - e2exp) + iwC*(w - w0)

    h        = Phi*w;
    e2exp    = (e.^2) .* exp(-h);                     % n x 1
    A        = 0.5 * (Phi' * (bsxfun(@times, Phi, e2exp))) + iwC; % q x q
    b        = 0.5 * (Phi' * (one_n - e2exp)) + iwC * (w - w0);   % q x 1

    % Solve for Newton step
    % Try chol; fall back to pcg
    use_chol = true;
    try
        R = chol((A + A')/2, 'lower');
        dw = R'\(R\b);
        % Posterior covariance approx:
        Cw = R'\(R\eye(q));
    catch
        use_chol = false;
        if opts.verbose, fprintf('[update_hyper] Cholesky failed, using PCG.\n'); end
        [dw,flag,relres] = pcg((A + A')/2, b, 1e-8, 500);
        if flag~=0 && opts.verbose
            fprintf('[update_hyper] PCG flag=%d relres=%.2e\n', flag, relres);
        end
        % crude covariance fallback (diagonal of inv(A)):
        Cw = diag(1./max(1e-10, diag(A)));
    end

    % propose update
    w_prop = w + dw;

    % optional backtracking to ensure ascent in L_h
    if opts.backtrack
        L_curr = L_before;
        step   = 1.0;
        accepted = false;
        for tr = 1:opts.maxTries
            tries_total = tries_total + 1;
            w_try = w + step * dw;
            L_try = L_obj(w_try);
            if L_try > L_curr
                w        = w_try;
                L_before = L_try;
                accepted = true;
                break
            else
                step = step * opts.stepScale;
            end
        end
        if ~accepted
            % no improvement; stop early
            if opts.verbose
                fprintf('[update_hyper] Backtracking failed to improve L_h.\n');
            end
            break
        end
    else
        % accept full step
        w = w_prop;
        L_before = L_obj(w);
    end
end

% Final quantities
h       = Phi * w;
sigma2  = exp(h);

% Ensure symmetric PD posterior covariance
if exist('R','var') && ~isempty(R) && use_chol
    Cw = R'\(R\eye(q));
else
    % refresh A at final w to compute Cw more faithfully
    e2exp   = (e.^2) .* exp(-h);
    A_final = 0.5 * (Phi' * (bsxfun(@times, Phi, e2exp))) + iwC;
    A_final = (A_final + A_final')/2;
    % try chol; else diagonal fallback
    try
        Rf = chol(A_final,'lower');
        Cw = Rf'\(Rf\eye(q));
    catch
        Cw = diag(1./max(1e-10, diag(A_final)));
    end
end

% SPM-style reporting term L3
% L3 = 0.5*log|iwC*Cw| - 0.5*(w-w0)' iwC (w-w0)
%   = 0.5*(log|iwC| + log|Cw|) - 0.5*quad
quad  = (w - w0)' * (iwC * (w - w0));
logdet_iwC = safe_logdet(iwC);
logdet_Cw  = safe_logdet(Cw);
L3    = 0.5*(logdet_iwC + logdet_Cw) - 0.5*quad;

% stats
stats = struct();
stats.A        = A;         % last linear system (if backtracking accepted, approx)
stats.b        = b;
stats.L_before = [];        % not keeping full history; can add if needed
stats.L_after  = L_before;
stats.tries    = tries_total;

end

% -------- helpers --------
function ld = safe_logdet(M)
% logdet via chol if PD; otherwise fall back to eig (with guarding)
M  = (M + M')/2;
ld = NaN;
try
    R  = chol(M,'lower');
    ld = 2*sum(log(diag(R)));
catch
    d  = eig(M);
    d  = max(d, 1e-16);
    ld = sum(log(d));
end
end
