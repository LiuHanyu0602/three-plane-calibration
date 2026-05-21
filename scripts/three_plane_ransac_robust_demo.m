clc;
clear;
close all;

rng(5);

%% ============================================================
%  Three-plane calibration with RANSAC robust plane extraction
%
%  Compare:
%  A) SVD plane fitting + closed-form intersection
%  B) SVD plane fitting + Huber convex origin optimization
%  C) RANSAC plane fitting + closed-form intersection
%  D) RANSAC plane fitting + Huber convex origin optimization
% =============================================================

%% Settings

num_points_per_board = 1200;
board_size_1 = 1.0;
board_size_2 = 0.8;

noise_sigma = 0.002;        % 2 mm Gaussian noise
outlier_ratio = 0.08;       % 8% outliers
outlier_magnitude = 0.08;   % 80 mm outlier displacement

ransac_iter = 500;
ransac_threshold = 0.008;    % 8 mm inlier threshold
min_inlier_ratio = 0.60;

huber_delta = 0.006;         % 6 mm Huber threshold
max_huber_iter = 30;

%% ============================================================
%  1. Ground-truth three-plane configuration
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
%  2. LiDAR pose relative to world
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

points_L_with_outliers = cell(3,1);
outlier_masks_true = cell(3,1);

for i = 1:3

    tangent_shift = [0.30 * randn; 0.30 * randn];

    P_W = sample_points_on_plane( ...
        nW(:,i), O_W, tangent_shift, ...
        num_points_per_board, board_size_1, board_size_2);

    P_L = (R_WL' * (P_W' - t_WL))';

    P_L_noisy = P_L + noise_sigma * randn(size(P_L));

    P_L_out = P_L_noisy;

    num_outliers = round(outlier_ratio * num_points_per_board);
    outlier_idx = randperm(num_points_per_board, num_outliers);

    true_outlier_mask = false(num_points_per_board, 1);
    true_outlier_mask(outlier_idx) = true;

    for k = 1:num_outliers
        idx = outlier_idx(k);

        displacement = outlier_magnitude * (2 * rand - 1);
        P_L_out(idx,:) = P_L_out(idx,:) + displacement * nL_true(:,i)';
    end

    points_L_with_outliers{i} = P_L_out;
    outlier_masks_true{i} = true_outlier_mask;
end

%% ============================================================
%  4. Method A/B: ordinary SVD plane fitting
% =============================================================

nL_svd = zeros(3,3);
dL_svd = zeros(3,1);
residual_svd_all = zeros(3,1);

for i = 1:3
    [n_fit, d_fit] = fit_plane_svd(points_L_with_outliers{i});

    if dot(n_fit, nL_true(:,i)) < 0
        n_fit = -n_fit;
        d_fit = -d_fit;
    end

    nL_svd(:,i) = n_fit;
    dL_svd(i) = d_fit;

    residual = points_L_with_outliers{i} * n_fit + d_fit;
    residual_svd_all(i) = mean(abs(residual)) * 1000;
end

[O_svd_closed, R_svd] = frame_from_three_planes(nL_svd, dL_svd);

O_svd_huber = origin_convex_huber( ...
    nL_svd, points_L_with_outliers, ...
    huber_delta, max_huber_iter);

%% ============================================================
%  5. Method C/D: RANSAC robust plane fitting
% =============================================================

nL_ransac = zeros(3,3);
dL_ransac = zeros(3,1);

inlier_masks = cell(3,1);
inlier_ratio_est = zeros(3,1);
residual_ransac_all = zeros(3,1);
residual_ransac_inlier = zeros(3,1);
normal_error_svd_deg = zeros(3,1);
normal_error_ransac_deg = zeros(3,1);

for i = 1:3

    P = points_L_with_outliers{i};

    [n_r, d_r, inlier_mask] = ransac_plane_fit( ...
        P, ransac_iter, ransac_threshold);

    if dot(n_r, nL_true(:,i)) < 0
        n_r = -n_r;
        d_r = -d_r;
    end

    nL_ransac(:,i) = n_r;
    dL_ransac(i) = d_r;
    inlier_masks{i} = inlier_mask;

    inlier_ratio_est(i) = sum(inlier_mask) / length(inlier_mask);

    residual_all = P * n_r + d_r;
    residual_ransac_all(i) = mean(abs(residual_all)) * 1000;
    residual_ransac_inlier(i) = mean(abs(residual_all(inlier_mask))) * 1000;

    normal_error_svd_deg(i) = acosd_clamped(dot(nL_svd(:,i), nL_true(:,i)));
    normal_error_ransac_deg(i) = acosd_clamped(dot(nL_ransac(:,i), nL_true(:,i)));
end

[O_ransac_closed, R_ransac] = frame_from_three_planes(nL_ransac, dL_ransac);

% For Huber origin optimization, use only RANSAC inliers
points_inliers = cell(3,1);
for i = 1:3
    points_inliers{i} = points_L_with_outliers{i}(inlier_masks{i}, :);
end

O_ransac_huber = origin_convex_huber( ...
    nL_ransac, points_inliers, ...
    huber_delta, max_huber_iter);

%% ============================================================
%  6. Evaluation
% =============================================================

fprintf('\n================ RANSAC Robust Plane Demo ================\n');
fprintf('Noise sigma: %.2f mm\n', noise_sigma * 1000);
fprintf('Outlier ratio: %.1f %%\n', outlier_ratio * 100);
fprintf('Outlier magnitude: %.1f mm\n', outlier_magnitude * 1000);
fprintf('RANSAC threshold: %.1f mm\n', ransac_threshold * 1000);

fprintf('\nPlane fitting residuals:\n');
for i = 1:3
    fprintf('Plane %d:\n', i);
    fprintf('  SVD mean abs residual on all points: %.4f mm\n', residual_svd_all(i));
    fprintf('  RANSAC mean abs residual on all points: %.4f mm\n', residual_ransac_all(i));
    fprintf('  RANSAC mean abs residual on inliers: %.4f mm\n', residual_ransac_inlier(i));
    fprintf('  RANSAC estimated inlier ratio: %.2f %%\n', inlier_ratio_est(i) * 100);
    fprintf('  SVD normal error: %.6f deg\n', normal_error_svd_deg(i));
    fprintf('  RANSAC normal error: %.6f deg\n', normal_error_ransac_deg(i));
end

evaluate_method('A. SVD + closed-form intersection', ...
    O_svd_closed, R_svd, O_L_true, R_LB_true);

evaluate_method('B. SVD + Huber convex origin', ...
    O_svd_huber, R_svd, O_L_true, R_LB_true);

evaluate_method('C. RANSAC + closed-form intersection', ...
    O_ransac_closed, R_ransac, O_L_true, R_LB_true);

evaluate_method('D. RANSAC + Huber convex origin', ...
    O_ransac_huber, R_ransac, O_L_true, R_LB_true);

%% ============================================================
%  7. Visualization
% =============================================================

figure;
hold on;
grid on;
axis equal;

colors = {'r', 'g', 'b'};

for i = 1:3
    P = points_L_with_outliers{i};
    inMask = inlier_masks{i};

    scatter3(P(inMask,1), P(inMask,2), P(inMask,3), ...
        8, colors{i}, 'filled');

    scatter3(P(~inMask,1), P(~inMask,2), P(~inMask,3), ...
        20, 'k', 'x');
end

plot_frame(O_L_true, R_LB_true, 0.40, 'True');
plot_frame(O_svd_closed, R_svd, 0.35, 'SVD');
plot_frame(O_ransac_closed, R_ransac, 0.30, 'RANSAC');

xlabel('X_L / m');
ylabel('Y_L / m');
zlabel('Z_L / m');
title('Robust three-plane calibration with RANSAC plane extraction');
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
    c = mean(P, 1);
    Q = P - c;

    [~, ~, V] = svd(Q, 0);

    n = V(:, end);
    n = n / norm(n);

    d = -n' * c';
end

function [n_best, d_best, inlier_best] = ransac_plane_fit(P, maxIter, threshold)
    N = size(P,1);

    best_count = 0;
    best_mean_res = inf;

    n_best = [0;0;1];
    d_best = 0;
    inlier_best = false(N,1);

    for iter = 1:maxIter

        idx = randperm(N, 3);

        p1 = P(idx(1),:)';
        p2 = P(idx(2),:)';
        p3 = P(idx(3),:)';

        v1 = p2 - p1;
        v2 = p3 - p1;

        n = cross(v1, v2);

        if norm(n) < 1e-10
            continue;
        end

        n = n / norm(n);
        d = -n' * p1;

        residual = abs(P * n + d);
        inlier = residual < threshold;

        count = sum(inlier);

        if count < 3
            continue;
        end

        mean_res = mean(residual(inlier));

        if count > best_count || ...
           (count == best_count && mean_res < best_mean_res)

            best_count = count;
            best_mean_res = mean_res;

            n_best = n;
            d_best = d;
            inlier_best = inlier;
        end
    end

    % Refit using all inliers
    if sum(inlier_best) >= 3
        [n_refit, d_refit] = fit_plane_svd(P(inlier_best,:));

        residual = abs(P * n_refit + d_refit);
        inlier_refit = residual < threshold;

        if sum(inlier_refit) >= 3
            [n_refit2, d_refit2] = fit_plane_svd(P(inlier_refit,:));

            n_best = n_refit2;
            d_best = d_refit2;
            inlier_best = inlier_refit;
        else
            n_best = n_refit;
            d_best = d_refit;
        end
    end
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

function O = origin_convex_huber(nMat, pointsCell, delta, maxIter)
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

function angle_deg = acosd_clamped(x)
    x = max(min(x, 1), -1);
    angle_deg = acos(x) * 180 / pi;
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