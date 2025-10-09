function stats = cv_peb_ard_novar(y, X, K, opts)
    if nargin<4, opts = struct; end
    N = size(X,1); d = size(y,2);
    if isvector(y), y = y(:); d=1; end
    fold = crossvalind('Kfold', N, K);
    yh   = nan(N,d);
    for k = 1:K
        tr = fold~=k; te = ~tr;
        M  = peb_ard_novar(y(tr,:), X(tr,:), opts);
        [yh(te,:), ~] = peb_ard_predict(X(te,:), M);
    end
    res = y - yh;
    stats.rmse = sqrt(mean(res.^2,'omitnan'));
    stats.mae  = mean(abs(res),'omitnan');
    stats.r2   = 1 - sum(res.^2,'omitnan')./sum( (y - mean(y)).^2,'omitnan');
    stats.yhat = yh;
end
