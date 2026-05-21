clc;
clear;
close all;

rng(6);

%% ============================================================
%  Monte Carlo: robustness against outlier ratio
%
%  Compare:
%  A) SVD + closed-form intersection
%  B) RANSAC + closed-form intersection
%  C) RANSAC + Huber convex origin
% =============================================================

num_trials = 100;

outlier_ratio_list = [0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20];

num_points_per_board = 1200;
board_size_1 = 1.0;
board_size_2 = 0.8;

noise_sigma = 0.002;        % 2 mm Gaussian noise
outlier_magnitude = 0.08;   % 80 mm outlier displacement

ransac_iter = 500;
ransac_threshold = 0.008;   % 8 mm
huber_delta = 0.006;        % 6 mm
max_huber_iter = 30;

%% ============================================================
%  Ground-truth three-plane configuration
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
%  Storage
% =============================================================

num_ratios = length(outlier_ratio_list);

origin_svd = zeros(num_trials, num_ratios);
origin_ransac = zeros(num_trials, num_ratios);
origin_ransac_huber = zeros(num_trials, num_ratios);

rot_svd = zeros(num_trials, num_ratios);
rot_ransac = zeros(num_trials, num_ratios);
rot_ransac_huber = zeros(num_trials, num_ratios);

inlier_ratio_mean = zeros(num_trials, num_ratios);

%% ============================================================
%  Main Monte Carlo loop
% =============================================================

for ridx = 1:num_ratios

    outlier_ratio = outlier_ratio_list(ridx);

    for trial = 1:num_trials

        points_L = cell(3,1);

        for i = 1:3

            tangent_shift = [0.30 * randn; 0.30 * randn];

            P_W = sample_points_on_plane( ...
                nW(:,i), O_W, tangent_shift, ...
                num_points_per_board, board_size_1, board_size_2);

            P_L = (R_WL' * (P_W' - t_WL))';

            P_L = P_L + noise_sigma * randn(size(P_L));

            % Add outliers along the true normal direction
            num_outliers = round(outlier_ratio * num_points_per_board);
            if num_outliers > 0
                outlier_idx = randperm(num_points_per_board, num_outliers);

                for k = 1:num_outliers
                    idx = outlier_idx(k);
                    displacement = outlier_magnitude * (2 * rand - 1);
                    P_L(idx,:) = P_L(idx,:) + displacement * nL_true(:,i)';
                end
            end

            points_L{i} = P_L;
        end

        %% ----------------------------------------------------
        % A) SVD plane fitting
        %% ----------------------------------------------------

        n_svd = zeros(3,3);
        d_svd = zeros(3,1);

        for i = 1:3
            [n_fit, d_fit] = fit_plane_svd(points_L{i});

            if dot(n_fit, nL_true(:,i)) < 0
                n_fit = -n_fit;
                d_fit = -d_fit;
            end

            n_svd(:,i) = n_fit;
            d_svd(i) = d_fit;
        end

        [O_svd, R_svd] = frame_from_three_planes(n_svd, d_svd);

        origin_svd(trial, ridx) = norm(O_svd - O_L_true) * 1000;
        rot_svd(trial, ridx) = rotation_error_deg(R_svd, R_LB_true);

        %% ----------------------------------------------------
        % B/C) RANSAC plane fitting
        %% ----------------------------------------------------

        n_ransac = zeros(3,3);
        d_ransac = zeros(3,1);
        inlier_masks = cell(3,1);

        for i = 1:3
            [n_r, d_r, inlier_mask] = ransac_plane_fit( ...
                points_L{i}, ransac_iter, ransac_threshold);

            if dot(n_r, nL_true(:,i)) < 0
                n_r = -n_r;
                d_r = -d_r;
            end

            n_ransac(:,i) = n_r;
            d_ransac(i) = d_r;
            inlier_masks{i} = inlier_mask;
        end

        [O_ransac, R_ransac] = frame_from_three_planes(n_ransac, d_ransac);

        origin_ransac(trial, ridx) = norm(O_ransac - O_L_true) * 1000;
        rot_ransac(trial, ridx) = rotation_error_deg(R_ransac, R_LB_true);

        %% ----------------------------------------------------
        % C) RANSAC + Huber origin using inliers
        %% ----------------------------------------------------

        points_inliers = cell(3,1);
        inlier_ratio_each = zeros(3,1);

        for i = 1:3
            points_inliers{i} = points_L{i}(inlier_masks{i}, :);
            inlier_ratio_each(i) = sum(inlier_masks{i}) / length(inlier_masks{i});
        end

        inlier_ratio_mean(trial, ridx) = mean(inlier_ratio_each);

        O_huber = origin_convex_huber( ...
            n_ransac, points_inliers, ...
            huber_delta, max_huber_iter);

        R_huber = R_ransac;

        origin_ransac_huber(trial, ridx) = norm(O_huber - O_L_true) * 1000;
        rot_ransac_huber(trial, ridx) = rotation_error_deg(R_huber, R_LB_true);
    end

    fprintf('Finished outlier ratio %.1f %%\n', outlier_ratio * 100);
end

%% ============================================================
%  Summary statistics
% =============================================================

mean_origin_svd = mean(origin_svd, 1);
mean_origin_ransac = mean(origin_ransac, 1);
mean_origin_huber = mean(origin_ransac_huber, 1);

p95_origin_svd = zeros(1, num_ratios);
p95_origin_ransac = zeros(1, num_ratios);
p95_origin_huber = zeros(1, num_ratios);

mean_rot_svd = mean(rot_svd, 1);
mean_rot_ransac = mean(rot_ransac, 1);
mean_rot_huber = mean(rot_ransac_huber, 1);

p95_rot_svd = zeros(1, num_ratios);
p95_rot_ransac = zeros(1, num_ratios);
p95_rot_huber = zeros(1, num_ratios);

for ridx = 1:num_ratios
    p95_origin_svd(ridx) = percentile_simple(origin_svd(:,ridx), 95);
    p95_origin_ransac(ridx) = percentile_simple(origin_ransac(:,ridx), 95);
    p95_origin_huber(ridx) = percentile_simple(origin_ransac_huber(:,ridx), 95);

    p95_rot_svd(ridx) = percentile_simple(rot_svd(:,ridx), 95);
    p95_rot_ransac(ridx) = percentile_simple(rot_ransac(:,ridx), 95);
    p95_rot_huber(ridx) = percentile_simple(rot_ransac_huber(:,ridx), 95);
end

summary_table = table( ...
    (outlier_ratio_list(:) * 100), ...
    mean(inlier_ratio_mean, 1)', ...
    mean_origin_svd(:), ...
    mean_origin_ransac(:), ...
    mean_origin_huber(:), ...
    p95_origin_svd(:), ...
    p95_origin_ransac(:), ...
    p95_origin_huber(:), ...
    mean_rot_svd(:), ...
    mean_rot_ransac(:), ...
    mean_rot_huber(:), ...
    p95_rot_svd(:), ...
    p95_rot_ransac(:), ...
    p95_rot_huber(:), ...
    'VariableNames', { ...
    'Outlier_Ratio_percent', ...
    'Mean_RANSAC_Inlier_Ratio', ...
    'Mean_Origin_SVD_mm', ...
    'Mean_Origin_RANSAC_mm', ...
    'Mean_Origin_RANSAC_Huber_mm', ...
    'P95_Origin_SVD_mm', ...
    'P95_Origin_RANSAC_mm', ...
    'P95_Origin_RANSAC_Huber_mm', ...
    'Mean_Rotation_SVD_deg', ...
    'Mean_Rotation_RANSAC_deg', ...
    'Mean_Rotation_RANSAC_Huber_deg', ...
    'P95_Rotation_SVD_deg', ...
    'P95_Rotation_RANSAC_deg', ...
    'P95_Rotation_RANSAC_Huber_deg'});

disp(' ');
disp('================ Outlier Monte Carlo Summary ================');
disp(summary_table);

%% ============================================================
%  Plots
% =============================================================

x = outlier_ratio_list * 100;

figure;
plot(x, mean_origin_svd, '-o', 'LineWidth', 2);
hold on;
plot(x, mean_origin_ransac, '-s', 'LineWidth', 2);
plot(x, mean_origin_huber, '-^', 'LineWidth', 2);
grid on;
xlabel('Outlier ratio / %');
ylabel('Mean origin error / mm');
title('Origin error under different outlier ratios');
legend('SVD + closed-form', 'RANSAC + closed-form', 'RANSAC + Huber', ...
    'Location', 'northwest');

figure;
plot(x, mean_rot_svd, '-o', 'LineWidth', 2);
hold on;
plot(x, mean_rot_ransac, '-s', 'LineWidth', 2);
plot(x, mean_rot_huber, '-^', 'LineWidth', 2);
grid on;
xlabel('Outlier ratio / %');
ylabel('Mean rotation error / degree');
title('Rotation error under different outlier ratios');
legend('SVD + closed-form', 'RANSAC + closed-form', 'RANSAC + Huber', ...
    'Location', 'northwest');

figure;
plot(x, p95_origin_svd, '-o', 'LineWidth', 2);
hold on;
plot(x, p95_origin_ransac, '-s', 'LineWidth', 2);
plot(x, p95_origin_huber, '-^', 'LineWidth', 2);
grid on;
xlabel('Outlier ratio / %');
ylabel('95th percentile origin error / mm');
title('95th percentile origin error under outliers');
legend('SVD + closed-form', 'RANSAC + closed-form', 'RANSAC + Huber', ...
    'Location', 'northwest');

figure;
plot(x, p95_rot_svd, '-o', 'LineWidth', 2);
hold on;
plot(x, p95_rot_ransac, '-s', 'LineWidth', 2);
plot(x, p95_rot_huber, '-^', 'LineWidth', 2);
grid on;
xlabel('Outlier ratio / %');
ylabel('95th percentile rotation error / degree');
title('95th percentile rotation error under outliers');
legend('SVD + closed-form', 'RANSAC + closed-form', 'RANSAC + Huber', ...
    'Location', 'northwest');

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

function err = rotation_error_deg(R_est, R_true)
    R_err = R_est' * R_true;
    value = (trace(R_err) - 1) / 2;
    value = max(min(value, 1), -1);
    err = acos(value) * 180 / pi;
end

function p = percentile_simple(x, percent)
    x = sort(x(:));
    n = length(x);

    if n == 1
        p = x;
        return;
    end

    rank = 1 + (percent / 100) * (n - 1);
    low = floor(rank);
    high = ceil(rank);

    if low == high
        p = x(low);
    else
        weight = rank - low;
        p = (1 - weight) * x(low) + weight * x(high);
    end
end