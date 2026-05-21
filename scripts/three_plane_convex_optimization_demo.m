clc;
clear;
close all;

rng(4);

%% ============================================================
%  Three-plane calibration with convex origin optimization
%  Goal:
%  Compare:
%  1) closed-form intersection of fitted planes
%  2) point-level convex least squares
%  3) point-level robust Huber convex optimization
% =============================================================

%% Basic settings

num_points_per_board = 1200;
board_size_1 = 1.0;
board_size_2 = 0.8;

noise_sigma = 0.002;        % 2 mm Gaussian noise
outlier_ratio = 0.08;       % 8% outliers
outlier_magnitude = 0.08;   % outliers up to 8 cm along normal direction

huber_delta = 0.006;        % 6 mm Huber threshold
max_huber_iter = 30;

%% ============================================================
%  1. Define ground-truth three-plane configuration
% =============================================================

nW = zeros(3,3);
nW(:,1) = normalize_vec([1.0; 0.25; 0.10]);
nW(:,2) = normalize_vec([0.15; 1.0; 0.30]);
nW(:,3) = normalize_vec([0.20; -0.20; 1.0]);

O_W = [0.30; -0.20; 0.40];

dW = zeros(3,1);
for i = 1:3
    dW(i) = -nW(:,i)' * O_W;
end

[O_W_check, R_WB] = frame_from_three_planes(nW, dW);

%% ============================================================
%  2. Define LiDAR pose relative to world
%     p_W = R_WL * p_L + t_WL
% =============================================================

yaw   = deg2rad(35);
pitch = deg2rad(-10);
roll  = deg2rad(8);

R_WL = eulzyx(yaw, pitch, roll);
t_WL = [1.20; -0.60; 0.80];

O_L_true = R_WL' * (O_W_check - t_WL);
R_LB_true = R_WL' * R_WB;

%% True plane parameters in LiDAR frame

nL_true = zeros(3,3);
dL_true = zeros(3,1);

for i = 1:3
    nL_true(:,i) = R_WL' * nW(:,i);
    dL_true(i) = nW(:,i)' * t_WL + dW(i);
end

%% ============================================================
%  3. Generate noisy point clouds with outliers
% =============================================================

points_L_clean = cell(3,1);
points_L_noisy = cell(3,1);
points_L_with_outliers = cell(3,1);

for i = 1:3

    tangent_shift = [0.30 * randn; 0.30 * randn];

    P_W = sample_points_on_plane( ...
        nW(:,i), O_W, tangent_shift, ...
        num_points_per_board, board_size_1, board_size_2);

    P_L = (R_WL' * (P_W' - t_WL))';

    P_L_noisy = P_L + noise_sigma * randn(size(P_L));

    % Add outliers mainly along plane normal direction
    P_L_out = P_L_noisy;

    num_outliers = round(outlier_ratio * num_points_per_board);
    outlier_idx = randperm(num_points_per_board, num_outliers);

    for k = 1:num_outliers
        idx = outlier_idx(k);

        % random displacement along the true plane normal
        displacement = outlier_magnitude * (2 * rand - 1);

        P_L_out(idx,:) = P_L_out(idx,:) + displacement * nL_true(:,i)';
    end

    points_L_clean{i} = P_L;
    points_L_noisy{i} = P_L_noisy;
    points_L_with_outliers{i} = P_L_out;
end

%% ============================================================
%  4. Fit planes using simple SVD
%     This intentionally shows how outliers can affect plane fitting.
% =============================================================

nL_svd = zeros(3,3);
dL_svd = zeros(3,1);
residual_svd = zeros(3,1);

for i = 1:3
    [n_fit, d_fit] = fit_plane_svd(points_L_with_outliers{i});

    if dot(n_fit, nL_true(:,i)) < 0
        n_fit = -n_fit;
        d_fit = -d_fit;
    end

    nL_svd(:,i) = n_fit;
    dL_svd(i) = d_fit;

    residual = points_L_with_outliers{i} * n_fit + d_fit;
    residual_svd(i) = mean(abs(residual)) * 1000;
end

%% ============================================================
%  5. Method A: closed-form intersection from fitted planes
% =============================================================

[O_closed, R_closed] = frame_from_three_planes(nL_svd, dL_svd);

%% ============================================================
%  6. Method B: point-level convex least squares
%
%  min_O sum_{i,j} (n_i^T (O - p_ij))^2
%
%  This is an unconstrained convex quadratic problem.
%  It can be solved by normal equations:
%
%  A O = b
% =============================================================

O_lsq = origin_convex_lsq(nL_svd, points_L_with_outliers);

R_lsq = axes_from_normals(nL_svd);

%% ============================================================
%  7. Method C: point-level robust Huber convex optimization
%
%  min_O sum rho_delta(n_i^T(O-p_ij))
%
%  Solved by iterative reweighted least squares.
% =============================================================

O_huber = origin_convex_huber( ...
    nL_svd, points_L_with_outliers, ...
    huber_delta, max_huber_iter);

R_huber = axes_from_normals(nL_svd);

%% ============================================================
%  8. Evaluation
% =============================================================

fprintf('\n================ Convex Optimization Demo ================\n');
fprintf('Noise sigma: %.2f mm\n', noise_sigma * 1000);
fprintf('Outlier ratio: %.1f %%\n', outlier_ratio * 100);
fprintf('Outlier magnitude: %.1f mm\n', outlier_magnitude * 1000);

fprintf('\nPlane fitting residuals using SVD on outlier-contaminated points:\n');
for i = 1:3
    fprintf('Plane %d mean abs residual: %.4f mm\n', i, residual_svd(i));
end

evaluate_method('A. Closed-form intersection', ...
    O_closed, R_closed, O_L_true, R_LB_true);

evaluate_method('B. Point-level convex LS', ...
    O_lsq, R_lsq, O_L_true, R_LB_true);

evaluate_method('C. Point-level Huber convex optimization', ...
    O_huber, R_huber, O_L_true, R_LB_true);

%% ============================================================
%  9. Visualization
% =============================================================

figure;
hold on;
grid on;
axis equal;

colors = {'r', 'g', 'b'};

for i = 1:3
    scatter3(points_L_with_outliers{i}(:,1), ...
             points_L_with_outliers{i}(:,2), ...
             points_L_with_outliers{i}(:,3), ...
             8, colors{i}, 'filled');
end

plot_frame(O_L_true, R_LB_true, 0.4, 'True');
plot_frame(O_closed, R_closed, 0.35, 'Closed');
plot_frame(O_lsq, R_lsq, 0.30, 'LS');
plot_frame(O_huber, R_huber, 0.25, 'Huber');

xlabel('X_L / m');
ylabel('Y_L / m');
zlabel('Z_L / m');
title('Convex origin optimization for three-plane calibration');
view(35, 25);

legend('Plane 1', 'Plane 2', 'Plane 3');

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
    c = mean(P, 1);
    Q = P - c;

    [~, ~, V] = svd(Q, 0);

    n = V(:, end);
    n = n / norm(n);

    d = -n' * c';
end

function [O, R] = frame_from_three_planes(nMat, dVec)
    N = nMat';
    O = N \ (-dVec(:));
    R = axes_from_normals(nMat);
end

function R = axes_from_normals(nMat)
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

    if dot(e3, n3) < 0
        e2 = -e2;
        e3 = cross(e1, e2);
        e3 = e3 / norm(e3);
    end

    R = [e1, e2, e3];
end

function O = origin_convex_lsq(nMat, pointsCell)
    % Solve:
    % min_O sum_{i,j} (n_i^T (O - p_ij))^2
    %
    % This is equivalent to:
    % min_O ||A O - b||^2

    A = [];
    b = [];

    for i = 1:3
        n = nMat(:,i);
        P = pointsCell{i};

        Ni = size(P,1);

        Ai = repmat(n', Ni, 1);
        bi = P * n;

        A = [A; Ai];
        b = [b; bi];
    end

    O = (A' * A) \ (A' * b);
end

function O = origin_convex_huber(nMat, pointsCell, delta, maxIter)
    % Robust convex Huber optimization using IRLS.
    %
    % Objective:
    % min_O sum rho_delta(n_i^T(O - p_ij))
    %
    % Huber weight:
    % w = 1,                      if |r| <= delta
    % w = delta / |r|,             if |r| > delta

    A = [];
    b = [];

    for i = 1:3
        n = nMat(:,i);
        P = pointsCell{i};

        Ni = size(P,1);

        Ai = repmat(n', Ni, 1);
        bi = P * n;

        A = [A; Ai];
        b = [b; bi];
    end

    % initialize using LS
    O = (A' * A) \ (A' * b);

    for iter = 1:maxIter

        r = A * O - b;

        abs_r = abs(r);
        w = ones(size(r));

        idx = abs_r > delta;
        w(idx) = delta ./ abs_r(idx);

        Wsqrt = sqrt(w);

        Aw = A .* Wsqrt;
        bw = b .* Wsqrt;

        O_new = (Aw' * Aw) \ (Aw' * bw);

        if norm(O_new - O) < 1e-10
            O = O_new;
            break;
        end

        O = O_new;
    end
end

function evaluate_method(name, O_est, R_est, O_true, R_true)
    origin_error_mm = norm(O_est - O_true) * 1000;

    R_err = R_est' * R_true;
    rotation_error_deg = rotation_angle_deg(R_err);

    fprintf('\n%s\n', name);
    fprintf('Origin error: %.4f mm\n', origin_error_mm);
    fprintf('Rotation error: %.6f deg\n', rotation_error_deg);
end

function angle_deg = rotation_angle_deg(R)
    value = (trace(R) - 1) / 2;
    value = max(min(value, 1), -1);
    angle_deg = acos(value) * 180 / pi;
end

function plot_frame(O, R, scale, nameStr)
    quiver3(O(1), O(2), O(3), ...
        scale*R(1,1), scale*R(2,1), scale*R(3,1), ...
        'LineWidth', 2, 'Color', 'r');

    quiver3(O(1), O(2), O(3), ...
        scale*R(1,2), scale*R(2,2), scale*R(3,2), ...
        'LineWidth', 2, 'Color', 'g');

    quiver3(O(1), O(2), O(3), ...
        scale*R(1,3), scale*R(2,3), scale*R(3,3), ...
        'LineWidth', 2, 'Color', 'b');

    plot3(O(1), O(2), O(3), ...
        'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'k');

    text(O(1), O(2), O(3), ['  ', nameStr], ...
        'FontSize', 10, 'FontWeight', 'bold');
end