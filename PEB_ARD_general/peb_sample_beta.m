function Bsamples = peb_sample_beta(M, nsamp)
% Returns nsamp×p matrix (original scale) for d=1. Extend to d>1 if needed.
    if nargin<2, nsamp=2000; end
    p = size(M.beta_std,1);
    % Chol of Vbeta (std space)
    [R,flag] = chol(M.Vbeta, 'lower');
    if flag~=0
        % repair if needed
        [U,S] = svd((M.Vbeta+M.Vbeta')/2);
        s = max(diag(S),1e-10);
        L = U*diag(sqrt(s));
    else
        L = R;
    end
    Z = randn(p, nsamp);
    Bstd = M.beta_std + L*Z;                  % p×nsamp (std)
    scale = (M.y_std(1) ./ M.x_std(:));       % p×1
    Bsamples = (scale .* Bstd).';             % nsamp×p (orig)
end

% Example density ribbon for one coefficient (say X2):
% Bsamp = peb_sample_beta(M, 5000);
% figure; histogram(Bsamp(:,3), 60, 'Normalization','pdf'); grid on
% title('Posterior samples for \beta_{X2}')
% xlabel('\beta'); ylabel('density')
