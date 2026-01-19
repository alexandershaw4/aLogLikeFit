function S = plotPolyphonicPosterior(OUT, idx, NS, OPT)
% Visualise multimodal posterior from Polyphonic VL (mixture of Gaussians)
%
% OUT: output of fitVariationalLaplaceThermoPolyphonic
% idx: parameter indices to visualise (e.g., 1:6 or [2 5 9])
% NS : number of samples from mixture (e.g., 5000)
% OPT fields (optional):
%   .maxKShow     : how many components to show individually (default all)
%   .corner       : if true, make corner-style grid (default true if numel(idx)<=6)
%   .bins         : histogram bins (default 60)
%   .showKDE      : show KDE line (default false; keeps it toolbox-free)
%
% Returns S struct with samples and component assignments.

if nargin < 2 || isempty(idx), idx = 1:min(6,numel(OUT.voices(1).m)); end
if nargin < 3 || isempty(NS),  NS  = 5000; end
if nargin < 4, OPT = struct(); end

bins      = getOpt(OPT,'bins',60);
maxKShow  = getOpt(OPT,'maxKShow',numel(OUT.voices));
corner    = getOpt(OPT,'corner', numel(idx) <= 6);

voices = OUT.voices;
pi_k   = OUT.pi(:);
K      = numel(voices);
d      = numel(voices(1).m);

% ---- build covariances from precisions ----
Sig = cell(K,1);
mu  = zeros(d,K);

for k = 1:K
    mu(:,k) = voices(k).m(:);
    L = voices(k).L_H;                   % chol(precision)
    I = eye(d);
    X = L \ I;
    Sig{k} = (L' \ X);                   % inv(H)
end

% ---- sample mixture ----
% component assignments
cdf = cumsum(pi_k / sum(pi_k));
u   = rand(NS,1);
z   = arrayfun(@(r) find(cdf>=r,1,'first'), u);

X = zeros(NS, numel(idx));
for s = 1:NS
    k = z(s);
    % sample from N(mu_k, Sig_k) using chol(Sig_k)
    % (force PD just in case)
    Sk = Sig{k};
    Sk = (Sk + Sk')/2;
    [C, p] = chol(Sk, 'lower');
    if p ~= 0
        Sk = Sk + 1e-6*eye(size(Sk));
        C  = chol(Sk, 'lower');
    end
    xs = mu(:,k) + C * randn(d,1);
    X(s,:) = xs(idx);
end

S.samples = X;
S.comp    = z;

% ---- plots ----
if corner
    figure('Position',[200 200 1100 900]); clf;
    p = numel(idx);

    for i = 1:p
        for j = 1:p
            subplot(p,p,(i-1)*p+j);
            if i == j
                % 1D mixture histogram
                histogram(X(:,i), bins, 'Normalization','pdf');
                hold on;

                % overlay component means as ticks (optionally)
                kk = 1:min(K,maxKShow);
                for k = kk
                    xmk = mu(idx(i),k);
                    yl = ylim;
                    plot([xmk xmk], [0 yl(2)*0.15], 'LineWidth',1);
                end
                hold off;
                axis tight; box off;
                if i < p, set(gca,'XTickLabel',[]); end
                set(gca,'YTick',[]);
            elseif i > j
                % pairwise scatter (downsample for speed)
                ss = min(NS, 2000);
                ii = randperm(NS, ss);
                scatter(X(ii,j), X(ii,i), 6, z(ii), 'filled');
                box off;
                if i < p, set(gca,'XTickLabel',[]); end
                if j > 1, set(gca,'YTickLabel',[]); end
            else
                axis off;
            end
        end
    end
    sgtitle('Polyphonic posterior: mixture samples + component markers');
else
    % Simple 1D marginals only
    figure('Position',[200 200 1100 300]); clf;
    p = numel(idx);
    for i = 1:p
        subplot(1,p,i);
        histogram(X(:,i), bins, 'Normalization','pdf');
        hold on;
        kk = 1:min(K,maxKShow);
        for k = kk
            xmk = mu(idx(i),k);
            yl = ylim;
            plot([xmk xmk], [0 yl(2)*0.15], 'LineWidth',1);
        end
        hold off;
        title(sprintf('param %d', idx(i)));
        box off; set(gca,'YTick',[]);
    end
    sgtitle('Polyphonic posterior: 1D mixture marginals');
end

end

function v = getOpt(S, name, default)
if isfield(S,name) && ~isempty(S.(name))
    v = S.(name);
else
    v = default;
end
end
