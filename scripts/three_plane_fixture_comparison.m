clc;
clear;
close all;

rng(7);

%% ============================================================
%  Comparison between standard fixture and arbitrary three-plane boards
%
%  Baselines:
%  1) Standard orthogonal trihedral fixture
%  2) Arbitrary well-conditioned three-plane boards
%  3) Arbitrary medium-conditioned three-plane boards
%  4) Arbitrary near-degenerate three-plane boards
%
%  All methods use:
%  RANSAC plane extraction + closed-form origin + axes from normals
% =============================================================

num_trials = 200;

num_points_per_board = 1200;
board_size_1 = 1.0;
board_size_2 = 0.8;

noise_sigma = 0.002;        % 2 mm Gaussian noise
outlier_ratio = 0.08;       % 8% outliers
outlier_magnitude = 0.08;   % 80 mm

ransac_iter = 500;
ransac_threshold = 0.008;   % 8 mm

O_W = [0.30; -0.20; 0.40];

yaw   = deg2rad(35);
pitch = deg2rad(-10);
roll  = deg2rad(8);

R_WL = eulzyx(yaw, pitch, roll);
t_WL = [1.20; -0.60; 0.80];

%% ============================================================
%  Define four geometry cases
% =============================================================

case_names = { ...
    'Standard orthogonal fixture', ...
    'Arbitrary well-conditioned boards', ...
    'Arbitrary medium-conditioned boards', ...
    'Arbitrary near-degenerate boards'};

num_cases = length(case_names);

nW_cases = cell(num_cases, 1);

% Case 1: ideal standard orthogonal trihedral fixture
nW1 = eye(3);
nW_cases{1} = nW1;

% Case 2: arbitrary but well-conditioned
nW2 = zeros(3,3);
nW2(:,1) = normalize_vec([1.0; 0.25; 0.10]);
nW2(:,2) = normalize_vec([0.15; 1.0; 0.30]);
nW2(:,3) = normalize_vec([0.20; -0.20; 1.0]);
nW_cases{2} = nW2;

% Case 3: medium-conditioned
theta = deg2rad(20);
nW3 = zeros(3,3);
nW3(:,1) = [1; 0; 0];
nW3(:,2) = [cos(theta); sin(theta); 0];
nW3(:,3) = normalize_vec([0.20; -0.30; 1.0]);
nW_cases{3} = nW3;

% Case 4: near-degenerate
theta = deg2rad(7);
nW4 = zeros(3,3);
nW4(:,1) = [1; 0; 0];
nW4(:,2) = [cos(theta); sin(theta); 0];
nW4(:,3) = normalize_vec([0.20; -0.30; 1.0]);
nW_cases{4} = nW4;

%% ============================================================
%  Storage
% =============================================================

origin_error = zeros(num_trials, num_cases);
rotation_error = zeros(num_trials, num_cases);
condition_number = zeros(num_cases, 1);
inlier_ratio_mean = zeros(num_trials, num_cases);

%% ============================================================
%  Main loop
% =============================================================

for cidx = 1:num_cases

    nW = nW_cases{cidx};

    for i = 1:3
        nW(:,i) = normalize_vec(nW(:,i));
    end

    dW = zeros(3,1);
    for i = 1:3
        dW(i) = -nW(:,i)' * O_W;
    end

    condition_number(cidx) = cond(nW');

    [O_W_check, R_WB] = frame_from_three_planes(nW, dW);

    O_L_true = R_WL' * (O_W_check - t_WL);
    R_LB_true = R_WL' * R_WB;

    nL_true = zeros(3,3);
    dL_true = zeros(3,1);

    for i = 1:3
        nL_true(:,i) = R_WL' * nW(:,i);
        dL_true(i) = nW(:,i)' * t_WL + dW(i);
    end

    for trial = 1:num_trials

        points_L = cell(3,1);

        for i = 1:3

            tangent_shift = [0.30 * randn; 0.30 * randn];

            P_W = sample_points_on_plane( ...
                nW(:,i), O_W, tangent_shift, ...
                num_points_per_board, board_size_1, board_size_2);

            P_L = (R_WL' * (P_W' - t_WL))';

            P_L = P_L + noise_sigma * randn(size(P_L));

            % Add outliers along normal direction
            num_outliers = round(outlier_ratio * num_points_per_board);
            outlier_idx = randperm(num_points_per_board, num_outliers);

            for k = 1:num_outliers
                idx = outlier_idx(k);
                displacement = outlier_magnitude * (2 * rand - 1);
                P_L(idx,:) = P_L(idx,:) + displacement * nL_true(:,i)';
            end

            points_L{i} = P_L;
        end

        % RANSAC fit
        n_ransac = zeros(3,3);
        d_ransac = zeros(3,1);
        inlier_ratio_each = zeros(3,1);

        for i = 1:3
            [n_r, d_r, inlier_mask] = ransac_plane_fit( ...
                points_L{i}, ransac_iter, ransac_threshold);

            if dot(n_r, nL_true(:,i)) < 0
                n_r = -n_r;
                d_r = -d_r;
            end

            n_ransac(:,i) = n_r;
            d_ransac(i) = d_r;
            inlier_ratio_each(i) = sum(inlier_mask) / length(inlier_mask);
        end

        inlier_ratio_mean(trial, cidx) = mean(inlier_ratio_each);

        [O_est, R_est] = frame_from_three_planes(n_ransac, d_ransac);

        origin_error(trial, cidx) = norm(O_est - O_L_true) * 1000;
        rotation_error(trial, cidx) = rotation_error_deg(R_est, R_LB_true);
    end

    fprintf('Finished case %d: %s, cond(N)=%.2f\n', ...
        cidx, case_names{cidx}, condition_number(cidx));
end

%% ============================================================
%  Summary
% =============================================================

mean_origin = mean(origin_error, 1);
p95_origin = zeros(1, num_cases);
std_origin = std(origin_error, 0, 1);

mean_rot = mean(rotation_error, 1);
p95_rot = zeros(1, num_cases);
std_rot = std(rotation_error, 0, 1);

for cidx = 1:num_cases
    p95_origin(cidx) = percentile_simple(origin_error(:,cidx), 95);
    p95_rot(cidx) = percentile_simple(rotation_error(:,cidx), 95);
end

summary_table = table( ...
    case_names(:), ...
    condition_number(:), ...
    mean(inlier_ratio_mean, 1)', ...
    mean_origin(:), ...
    std_origin(:), ...
    p95_origin(:), ...
    mean_rot(:), ...
    std_rot(:), ...
    p95_rot(:), ...
    'VariableNames', { ...
    'Geometry_Case', ...
    'Condition_Number', ...
    'Mean_RANSAC_Inlier_Ratio', ...
    'Mean_Origin_Error_mm', ...
    'Std_Origin_Error_mm', ...
    'P95_Origin_Error_mm', ...
    'Mean_Rotation_Error_deg', ...
    'Std_Rotation_Error_deg', ...
    'P95_Rotation_Error_deg'});

disp(' ');
disp('================ Fixture Comparison Summary ================');
disp(summary_table);

%% ============================================================
%  Plot
% =============================================================

figure;
bar(mean_origin);
hold on;
errorbar(1:num_cases, mean_origin, std_origin, 'k.', 'LineWidth', 1.5);
grid on;
set(gca, 'XTickLabel', case_names);
xtickangle(25);
ylabel('Mean origin error / mm');
title('Origin error: standard fixture vs arbitrary three-plane boards');

figure;
bar(mean_rot);
hold on;
errorbar(1:num_cases, mean_rot, std_rot, 'k.', 'LineWidth', 1.5);
grid on;
set(gca, 'XTickLabel', case_names);
xtickangle(25);
ylabel('Mean rotation error / degree');
title('Rotation error: standard fixture vs arbitrary three-plane boards');

figure;
bar(condition_number);
grid on;
set(gca, 'XTickLabel', case_names);
xtickangle(25);
ylabel('Condition number');
title('Geometry condition number of different board configurations');

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