function plotVLtrajectory3D(loss_fn, allm, m0, paramIdx, rangeFactor, numGrid)
% 3D Visualisation of error landscape and Variational Laplace trajectory
%
% Inputs:
%   loss_fn     - Function handle: takes m, returns scalar loss
%   allm        - Matrix of parameter vectors over iterations (columns)
%   m0          - Initial parameter vector
%   paramIdx    - Indices of 2 parameters to vary (e.g. [1 2])
%   rangeFactor - +/- range in std devs for axis scaling
%   numGrid     - Grid resolution (default = 50)

if nargin < 5, rangeFactor = 3; end
if nargin < 6, numGrid = 50; end

i = paramIdx(1);
j = paramIdx(2);

% Use the trajectory to define grid ranges
x_vals = linspace(min(allm(i,:)), max(allm(i,:)), numGrid);
y_vals = linspace(min(allm(j,:)), max(allm(j,:)), numGrid);
[X, Y] = meshgrid(x_vals, y_vals);
Z = zeros(size(X));

% Evaluate loss on grid
for r = 1:numGrid
    for c = 1:numGrid
        m_test = m0;
        m_test(i) = X(r,c);
        m_test(j) = Y(r,c);
        Z(r,c) = loss_fn(m_test);
    end
end

% Plot 3D surface
figure;
surf(X, Y, Z, 'EdgeColor', 'none', 'FaceAlpha', 0.9);
colormap turbo;
colorbar;
hold on;

% Overlay VL trajectory in 3D (Z = loss along trajectory)
traj_z = zeros(1, size(allm,2));
for k = 1:length(traj_z)
    traj_z(k) = loss_fn(allm(:,k));
end
plot3(allm(i,:), allm(j,:), traj_z, 'w-o', ...
    'LineWidth', 2, 'MarkerSize', 4, 'MarkerFaceColor', 'w', ...
    'DisplayName', 'VL Trajectory');

scatter3(allm(i,1), allm(j,1), traj_z(1), 60, 'g', 'filled', 'DisplayName', 'Start');
scatter3(allm(i,end), allm(j,end), traj_z(end), 60, 'r', 'filled', 'DisplayName', 'End');

xlabel(['Param ', num2str(i)]);
ylabel(['Param ', num2str(j)]);
zlabel('Loss');
title('3D Error Landscape with VL Trajectory');
legend;
grid on;
view(45, 30);  % adjustable viewing angle
axis tight;
end
