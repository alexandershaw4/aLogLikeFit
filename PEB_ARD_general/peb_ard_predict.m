function [yhat, ysd] = peb_ard_predict(Xnew, M)
% Posterior predictive mean and std on ORIGINAL scale.
% - Respects column pruning and training standardisation.

    % select kept columns
    Xn = Xnew(:, M.kept_columns);
    % standardise like training
    Xs = (Xn - M.x_mean) ./ M.x_std;
    % if there was an intercept col among kept, keep it as 1 exactly
    is_one = all(abs(Xs - 1) < 1e-12, 1);
    Xs(:, is_one) = 1;

    % mean in standardised space
    yhat_std = Xs * M.beta_std;              % Q×d

    % predictive variance diag: Var = σ²_std + x Vβ x'
    Q = size(Xs,1); d = size(M.beta_std,2);
    yvar_std = zeros(Q,d);
    for i = 1:Q
        xv = Xs(i,:).';
        xb = xv.' * M.Vbeta * xv;            % scalar
        yvar_std(i,:) = (M.sigma2 ./ (M.y_std.^2)) + xb;
    end

    % back to original scale
    yhat = yhat_std .* M.y_std + M.y_mean;
    ysd  = sqrt(max(yvar_std, 0)) .* M.y_std;
end
