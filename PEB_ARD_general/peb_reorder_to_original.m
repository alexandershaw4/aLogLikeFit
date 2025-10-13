function M2 = peb_reorder_to_original(M, p0)
%PEB_REORDER_TO_ORIGINAL  Reorder/expand PEB-ARD outputs to original X column order.
%
%   M2 = peb_reorder_to_original(M, p0)
%
% Takes the fitted model struct M (from peb_ard_novar) and the original number
% of predictors p0 (before pruning). Expands/reorders all relevant fields to
% align with the original column order of X.
%
% Dropped columns are filled with NaN (for vectors) or NaN/0 (for matrices).
%
% Example:
%   M2 = peb_reorder_to_original(M, size(X,2));
%   disp(M2.beta_ordered)

kept = M.kept_columns(:);
p_kept = numel(kept);

% Order kept columns ascending by their original index
[kept_sorted, ord_idx] = sort(kept);
P = eye(p_kept);
P = P(ord_idx, :);   % permute current->original order among kept

% --- Helper functions ---
    function v_out = expand_vec(v)
        v_out = nan(p0, size(v,2));
        v_ord = P * v;
        v_out(kept_sorted,:) = v_ord;
    end

    function A_out = expand_mat(A)
        A_out = nan(p0);
        A_ord = P * A * P';
        A_out(kept_sorted, kept_sorted) = A_ord;
    end

% Copy M and add ordered fields
M2 = M;

% --- Vectors ---
if isfield(M,'beta'),        M2.beta_ordered        = expand_vec(M.beta);        end
if isfield(M,'beta_std'),    M2.beta_ordered_std    = expand_vec(M.beta_std);    end
if isfield(M,'lambda'),      M2.lambda_ordered      = expand_vec(M.lambda);      end
if isfield(M,'gamma'),       M2.gamma_ordered       = expand_vec(M.gamma);       end
if isfield(M,'z'),           M2.z_ordered           = expand_vec(M.z);           end
if isfield(M,'pvals'),       M2.pvals_ordered       = expand_vec(M.pvals);       end
if isfield(M,'x_mean')
    tmp = expand_vec(M.x_mean.');
    M2.x_mean_ordered = tmp.';
end
if isfield(M,'x_std')
    tmp = expand_vec(M.x_std.');
    M2.x_std_ordered = tmp.';
end

% --- Matrices ---
if isfield(M,'Vbeta'),       M2.Vbeta_ordered       = expand_mat(M.Vbeta);       end

% --- Masks and mapping info ---
mask = false(p0,1);
mask(kept_sorted) = true;
M2.original_mask = mask;
M2.permutation_P = P;              % permutation among kept cols
M2.kept_sorted   = kept_sorted;    % ascending original indices
end
