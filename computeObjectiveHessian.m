function H = computeObjectiveHessian(obj, p, d)
n = numel(p);
H = zeros(n,n);
f0 = obj(p);

for i = 1:n
    ei = zeros(n,1); ei(i) = 1;

    % diagonal
    fp = obj(p + d*ei);
    fm = obj(p - d*ei);
    H(i,i) = (fp - 2*f0 + fm) / d^2;

    for j = i+1:n
        ej = zeros(n,1); ej(j) = 1;

        fpp = obj(p + d*ei + d*ej);
        fpm = obj(p + d*ei - d*ej);
        fmp = obj(p - d*ei + d*ej);
        fmm = obj(p - d*ei - d*ej);

        H(i,j) = (fpp - fpm - fmp + fmm) / (4*d^2);
        H(j,i) = H(i,j);
    end
end
end