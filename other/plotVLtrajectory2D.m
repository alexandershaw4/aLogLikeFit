function plotVLtrajectory2D(loss_fn, allm, m0, paramIdx, rangeFactor, numGrid)
% Visualise the loss/error landscape over two parameters with VL trajectory
%
% Inputs:
%   loss_fn     - Function handle: takes parameter vector m, returns scalar loss
%   allm        - Matrix of parameter vectors over iterations (columns)
%   m0          - Initial parameter vector
%   paramIdx    - Indices of the 2 parameters to vary (e.g., [1 2])
%   rangeFactor - +/- range in std devs for each axis (e.g., 3)
%   numGrid     - Number of grid points per dimension (e.g., 50)

if nargin < 5, rangeFactor = 3; end
if nargin < 6, numGrid = 50; end

% Select parameters
i = paramIdx(1); j = paramIdx(2);

% Estimate parameter ranges based on allm trajectory
m_i_vals = linspace(min(allm(i,:)), max(allm(i,:)), numGrid);
m_j_vals = linspace(min(allm(j,:)), max(allm(j,:)), numGrid);

% Create meshgrid
[Mi, Mj] = meshgrid(m_i_vals, m_j_vals);
Z = zeros(size(Mi));

% Evaluate loss at each grid point
for r = 1:numGrid
    for c = 1:numGrid
        m_test = m0;
        m_test(i) = Mi(r, c);
        m_test(j) = Mj(r, c);
        Z(r, c) = loss_fn(m_test);
    end
end

% Plot contour map
figure;
contourf(Mi, Mj, Z, 30, 'LineColor', 'none');
colormap turbo;
colorbar;
xlabel(['Param ', num2str(i)]);
ylabel(['Param ', num2str(j)]);
title('Error Landscape with VL Trajectory');
hold on;

% Overlay trajectory
plot(allm(i,:), allm(j,:), 'w-o', 'LineWidth', 2, ...
     'MarkerSize', 4, 'MarkerFaceColor', 'w', 'DisplayName', 'VL Path');

scatter(allm(i,1), allm(j,1), 60, 'g', 'filled', 'DisplayName', 'Start');
scatter(allm(i,end), allm(j,end), 60, 'r', 'filled', 'DisplayName', 'End');
legend;
grid on;
axis tight;
end
