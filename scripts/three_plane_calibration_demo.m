clc;
clear;
close all;

rng(1);

%% ============================================================
%  1. Ground-truth: define three arbitrary non-degenerate planes
%     in a board/world coordinate system
% =============================================================

% Three arbitrary plane normals in world frame
nW = zeros(3,3);
nW(:,1) = normalize_vec([1.0; 0.25; 0.10]);
nW(:,2) = normalize_vec([0.15; 1.0; 0.30]);
nW(:,3) = normalize_vec([0.20; -0.20; 1.0]);

% True intersection point of the three infinite planes
O_W = [0.30; -0.20; 0.40];

% Plane equation in world frame:
% n_i^T x + d_i = 0
dW = zeros(3,1);
for i = 1:3
    dW(i) = -nW(:,i)' * O_W;
end

% Build the true board-defined coordinate frame
[O_W_check, R_WB] = frame_from_three_planes(nW, dW);

fprintf('Ground-truth board-frame origin in world frame:\n');
disp(O_W_check');

%% ============================================================
%  2. Define LiDAR pose relative to world
%     p_W = R_WL * p_L + t_WL
% =============================================================

yaw   = deg2rad(35);
pitch = deg2rad(-10);
roll  = deg2rad(8);

R_WL = eulzyx(yaw, pitch, roll);
t_WL = [1.20; -0.60; 0.80];

% True board frame expressed in LiDAR frame
O_L_true = R_WL' * (O_W_check - t_WL);
R_LB_true = R_WL' * R_WB;

%% ============================================================
%  3. Generate synthetic point clouds on three plane boards
% =============================================================

num_points_per_board = 1200;
board_size_1 = 1.0;
board_size_2 = 0.8;

noise_sigma = 0.002;   % 2 mm noise

points_L = cell(3,1);
points_W = cell(3,1);

for i = 1:3
    % Random tangent shift so that the board patch is not exactly centered
    % at the three-plane intersection point
    tangent_shift = [0.30 * randn; 0.30 * randn];

    P_W = sample_points_on_plane( ...
        nW(:,i), O_W, tangent_shift, ...
        num_points_per_board, board_size_1, board_size_2);

    % Transform world points into LiDAR frame
    P_L = (R_WL' * (P_W' - t_WL))';

    % Add measurement noise in LiDAR frame
    P_L = P_L + noise_sigma * randn(size(P_L));

    points_W{i} = P_W;
    points_L{i} = P_L;
end

%% ============================================================
%  4. Fit three planes from LiDAR point clouds
% =============================================================

nL_est = zeros(3,3);
dL_est = zeros(3,1);

nL_true = zeros(3,3);
dL_true = zeros(3,1);

for i = 1:3
    % True plane in LiDAR frame
    % World plane: nW^T pW + dW = 0
    % pW = R_WL pL + t_WL
    % => (R_WL^T nW)^T pL + nW^T t_WL + dW = 0
    nL_true(:,i) = R_WL' * nW(:,i);
    dL_true(i) = nW(:,i)' * t_WL + dW(i);

    % Fit plane from noisy LiDAR points
    [n_fit, d_fit] = fit_plane_svd(points_L{i});

    % For simulation evaluation, align estimated normal direction
    % with the ground-truth normal direction
    if dot(n_fit, nL_true(:,i)) < 0
        n_fit = -n_fit;
        d_fit = -d_fit;
    end

    nL_est(:,i) = n_fit;
    dL_est(i) = d_fit;
end

%% ============================================================
%  5. Estimate board-frame origin and axes in LiDAR frame
% =============================================================

[O_L_est, R_LB_est] = frame_from_three_planes(nL_est, dL_est);

%% ============================================================
%  6. Evaluate calibration accuracy
% =============================================================

origin_error_mm = norm(O_L_est - O_L_true) * 1000;

R_err = R_LB_est' * R_LB_true;
rotation_error_deg = rotation_angle_deg(R_err);

axis_error_deg = zeros(3,1);
for k = 1:3
    axis_error_deg(k) = acosd_clamped(dot(R_LB_est(:,k), R_LB_true(:,k)));
end

fprintf('\n================ Calibration Result ================\n');
fprintf('Origin error: %.4f mm\n', origin_error_mm);
fprintf('Overall rotation error: %.6f deg\n', rotation_error_deg);
fprintf('Axis-1 direction error: %.6f deg\n', axis_error_deg(1));
fprintf('Axis-2 direction error: %.6f deg\n', axis_error_deg(2));
fprintf('Axis-3 direction error: %.6f deg\n', axis_error_deg(3));

fprintf('\nPlane fitting residuals:\n');
for i = 1:3
    residual = points_L{i} * nL_est(:,i) + dL_est(i);
    fprintf('Plane %d mean abs residual: %.4f mm\n', ...
        i, mean(abs(residual)) * 1000);
end

%% ============================================================
%  7. Visualization
% =============================================================

figure;
hold on;
grid on;
axis equal;

colors = {'r', 'g', 'b'};

for i = 1:3
    scatter3(points_L{i}(:,1), points_L{i}(:,2), points_L{i}(:,3), ...
        8, colors{i}, 'filled');
end

% Estimated coordinate frame
scale = 0.4;

quiver3(O_L_est(1), O_L_est(2), O_L_est(3), ...
    scale*R_LB_est(1,1), scale*R_LB_est(2,1), scale*R_LB_est(3,1), ...
    'LineWidth', 3, 'Color', 'r');

quiver3(O_L_est(1), O_L_est(2), O_L_est(3), ...
    scale*R_LB_est(1,2), scale*R_LB_est(2,2), scale*R_LB_est(3,2), ...
    'LineWidth', 3, 'Color', 'g');

quiver3(O_L_est(1), O_L_est(2), O_L_est(3), ...
    scale*R_LB_est(1,3), scale*R_LB_est(2,3), scale*R_LB_est(3,3), ...
    'LineWidth', 3, 'Color', 'b');

plot3(O_L_est(1), O_L_est(2), O_L_est(3), ...
    'ko', 'MarkerSize', 10, 'MarkerFaceColor', 'k');

xlabel('X_L / m');
ylabel('Y_L / m');
zlabel('Z_L / m');
title('Three-plane calibration result in LiDAR coordinate frame');

legend('Plane 1', 'Plane 2', 'Plane 3', ...
    'Estimated X axis', 'Estimated Y axis', 'Estimated Z axis', ...
    'Estimated origin');

view(35, 25);

%% ============================================================
%  Local functions
% =============================================================

function v = normalize_vec(v)
    v = v / norm(v);
end

function R = eulzyx(yaw, pitch, roll)
    Rz = [cos(yaw), -sin(yaw), 0;
          sin(yaw),  cos(yaw), 0;
          0,         0,        1];

    Ry = [cos(pitch), 0, sin(pitch);
          0,          1, 0;
         -sin(pitch), 0, cos(pitch)];

    Rx = [1, 0,          0;
          0, cos(roll), -sin(roll);
          0, sin(roll),  cos(roll)];

    R = Rz * Ry * Rx;
end

function P = sample_points_on_plane(n, O, tangent_shift, N, size1, size2)
    n = n / norm(n);

    tmp = [1; 0; 0];
    if abs(dot(tmp, n)) > 0.9
        tmp = [0; 1; 0];
    end

    b1 = cross(n, tmp);
    b1 = b1 / norm(b1);

    b2 = cross(n, b1);
    b2 = b2 / norm(b2);

    center = O + tangent_shift(1) * b1 + tangent_shift(2) * b2;

    u = (rand(N,1) - 0.5) * size1;
    v = (rand(N,1) - 0.5) * size2;

    P = center' + u * b1' + v * b2';
end

function [n, d] = fit_plane_svd(P)
    % P is N x 3
    c = mean(P, 1);
    Q = P - c;

    [~, ~, V] = svd(Q, 0);

    n = V(:, end);
    n = n / norm(n);

    d = -n' * c';
end

function [O, R] = frame_from_three_planes(nMat, dVec)
    % nMat: 3 x 3 matrix, each column is one plane normal
    % dVec: 3 x 1 vector
    %
    % Plane equation:
    % n_i^T x + d_i = 0

    N = nMat';

    % Intersection point of three planes
    O = N \ (-dVec(:));

    % Construct an orthonormal right-handed coordinate frame
    e1 = nMat(:,1);
    e1 = e1 / norm(e1);

    n2 = nMat(:,2);
    n2 = n2 / norm(n2);

    e2 = n2 - dot(e1, n2) * e1;
    e2 = e2 / norm(e2);

    e3 = cross(e1, e2);
    e3 = e3 / norm(e3);

    n3 = nMat(:,3);
    n3 = n3 / norm(n3);

    % Use the third plane normal to fix the sign direction
    if dot(e3, n3) < 0
        e2 = -e2;
        e3 = cross(e1, e2);
        e3 = e3 / norm(e3);
    end

    R = [e1, e2, e3];
end

function angle_deg = rotation_angle_deg(R)
    value = (trace(R) - 1) / 2;
    value = max(min(value, 1), -1);
    angle_deg = acos(value) * 180 / pi;
end

function angle_deg = acosd_clamped(x)
    x = max(min(x, 1), -1);
    angle_deg = acos(x) * 180 / pi;
end