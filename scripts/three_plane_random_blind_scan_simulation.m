clc;
clear;
close all;

rng(8);

%% ============================================================
%  Random blind-scan simulation for three-plane LiDAR calibration
%
%  Goal:
%  Simulate randomly placed planar boards, perform RANSAC plane
%  extraction, compute the geometry condition number, and evaluate
%  calibration accuracy under different condition-number groups.
%
%  Pipeline:
%  Random three-plane generation
%      -> noisy + outlier point cloud generation
%      -> RANSAC plane extraction
%      -> estimated condition number
%      -> origin and axes recovery
%      -> accuracy evaluation
% =============================================================

%% Basic simulation settings

num_trials = 500;

num_points_per_board = 1200;
board_size_1 = 1.0;
board_size_2 = 0.8;

noise_sigma = 0.002;        % 2 mm Gaussian noise
outlier_ratio = 0.08;       % 8% outliers
outlier_magnitude = 0.08;   % 80 mm outlier displacement

ransac_iter = 500;
ransac_threshold = 0.008;   % 8 mm inlier threshold

%% LiDAR pose relative to global simulation frame

yaw   = deg2rad(35);
pitch = deg2rad(-10);
roll  = deg2rad(8);

R_WL = eulzyx(yaw, pitch, roll);
t_WL = [1.20; -0.60; 0.80];

%% Storage

origin_error = nan(num_trials, 1);
rotation_error = nan(num_trials, 1);

condition_true = nan(num_trials, 1);
condition_est = nan(num_trials, 1);

mean_inlier_ratio = nan(num_trials, 1);

success_flag = false(num_trials, 1);

%% ============================================================
%  Main random blind-scan simulation loop
% =============================================================

for trial = 1:num_trials

    %% --------------------------------------------------------
    % 1. Randomly generate three plane normals
    %% --------------------------------------------------------

    nW = zeros(3,3);

    for i = 1:3
        nW(:,i) = random_unit_vector();
    end

    % True condition number of the random three-plane configuration
    condition_true(trial) = cond(nW');

    % Random intersection point of the three infinite planes
    % This point becomes the board-defined world-frame origin.
    O_W = [0.30; -0.20; 0.40] + 0.30 * randn(3,1);

    dW = zeros(3,1);
    for i = 1:3
        dW(i) = -nW(:,i)' * O_W;
    end

    %% --------------------------------------------------------
    % 2. Construct true board-defined frame
    %% --------------------------------------------------------

    [O_W_check, R_WB] = frame_from_three_planes(nW, dW);

    O_L_true = R_WL' * (O_W_check - t_WL);
    R_LB_true = R_WL' * R_WB;

    %% --------------------------------------------------------
    % 3. True plane parameters in LiDAR frame
    %% --------------------------------------------------------

    nL_true = zeros(3,3);
    dL_true = zeros(3,1);

    for i = 1:3
        nL_true(:,i) = R_WL' * nW(:,i);
        dL_true(i) = nW(:,i)' * t_WL + dW(i);
    end

    %% --------------------------------------------------------
    % 4. Generate point clouds on three finite planar boards
    %% --------------------------------------------------------

    points_L = cell(3,1);

    for i = 1:3

        % The finite board patch is not necessarily centered at the
        % three-plane intersection point. This better mimics arbitrary
        % board placement.
        tangent_shift = [0.50 * randn; 0.50 * randn];

        P_W = sample_points_on_plane( ...
            nW(:,i), O_W, tangent_shift, ...
            num_points_per_board, board_size_1, board_size_2);

        % Transform world points into LiDAR frame
        P_L = (R_WL' * (P_W' - t_WL))';

        % Add Gaussian point cloud noise
        P_L = P_L + noise_sigma * randn(size(P_L));

        % Add outliers along the true plane normal direction
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

    %% --------------------------------------------------------
    % 5. RANSAC plane extraction from blind point clouds
    %% --------------------------------------------------------

    nL_est = zeros(3,3);
    dL_est = zeros(3,1);
    inlier_ratio_each = zeros(3,1);

    valid_trial = true;

    for i = 1:3

        [n_r, d_r, inlier_mask] = ransac_plane_fit( ...
            points_L{i}, ransac_iter, ransac_threshold);

        if sum(inlier_mask) < 20
            valid_trial = false;
            break;
        end

        % In simulation we align normal signs with ground truth for
        % fair error evaluation. In real experiments, this should be
        % replaced by a deterministic sign rule.
        if dot(n_r, nL_true(:,i)) < 0
            n_r = -n_r;
            d_r = -d_r;
        end

        nL_est(:,i) = n_r;
        dL_est(i) = d_r;

        inlier_ratio_each(i) = sum(inlier_mask) / length(inlier_mask);
    end

    if ~valid_trial
        continue;
    end

    %% --------------------------------------------------------
    % 6. Estimated condition number
    %% --------------------------------------------------------

    condition_est(trial) = cond(nL_est');

    %% --------------------------------------------------------
    % 7. Estimate board-frame origin and axes in LiDAR frame
    %% --------------------------------------------------------

    [O_L_est, R_LB_est] = frame_from_three_planes(nL_est, dL_est);

    %% --------------------------------------------------------
    % 8. Evaluation
    %% --------------------------------------------------------

    origin_error(trial) = norm(O_L_est - O_L_true) * 1000;
    rotation_error(trial) = rotation_error_deg(R_LB_est, R_LB_true);

    mean_inlier_ratio(trial) = mean(inlier_ratio_each);

    success_flag(trial) = true;

    if mod(trial, 50) == 0
        fprintf('Finished trial %d / %d\n', trial, num_trials);
    end
end

%% ============================================================
%  Remove invalid trials
% =============================================================

valid_idx = success_flag & ...
            isfinite(origin_error) & ...
            isfinite(rotation_error) & ...
            isfinite(condition_est);

origin_error = origin_error(valid_idx);
rotation_error = rotation_error(valid_idx);

condition_true = condition_true(valid_idx);
condition_est = condition_est(valid_idx);

mean_inlier_ratio = mean_inlier_ratio(valid_idx);

num_valid = length(origin_error);

fprintf('\nValid trials: %d / %d\n', num_valid, num_trials);

%% ============================================================
%  Group analysis by estimated condition number
% =============================================================

group_names = { ...
    'Good: cond < 3', ...
    'Acceptable: 3 <= cond < 6', ...
    'Risky: 6 <= cond < 10', ...
    'Bad: cond >= 10'};

group_lower = [0, 3, 6, 10];
group_upper = [3, 6, 10, inf];

num_groups = length(group_names);

sample_count = zeros(num_groups, 1);
sample_percent = zeros(num_groups, 1);

mean_cond_est = nan(num_groups, 1);
mean_cond_true = nan(num_groups, 1);

mean_origin = nan(num_groups, 1);
std_origin = nan(num_groups, 1);
p95_origin = nan(num_groups, 1);

mean_rot = nan(num_groups, 1);
std_rot = nan(num_groups, 1);
p95_rot = nan(num_groups, 1);

mean_inlier = nan(num_groups, 1);

for g = 1:num_groups

    idx = condition_est >= group_lower(g) & condition_est < group_upper(g);

    sample_count(g) = sum(idx);
    sample_percent(g) = 100 * sample_count(g) / num_valid;

    if sample_count(g) > 0
        mean_cond_est(g) = mean(condition_est(idx));
        mean_cond_true(g) = mean(condition_true(idx));

        mean_origin(g) = mean(origin_error(idx));
        std_origin(g) = std(origin_error(idx));
        p95_origin(g) = percentile_simple(origin_error(idx), 95);

        mean_rot(g) = mean(rotation_error(idx));
        std_rot(g) = std(rotation_error(idx));
        p95_rot(g) = percentile_simple(rotation_error(idx), 95);

        mean_inlier(g) = mean(mean_inlier_ratio(idx));
    end
end

group_summary_table = table( ...
    group_names(:), ...
    sample_count(:), ...
    sample_percent(:), ...
    mean_cond_est(:), ...
    mean_cond_true(:), ...
    mean_inlier(:), ...
    mean_origin(:), ...
    std_origin(:), ...
    p95_origin(:), ...
    mean_rot(:), ...
    std_rot(:), ...
    p95_rot(:), ...
    'VariableNames', { ...
    'Condition_Group', ...
    'Sample_Count', ...
    'Sample_Percent', ...
    'Mean_Est_Condition_Number', ...
    'Mean_True_Condition_Number', ...
    'Mean_RANSAC_Inlier_Ratio', ...
    'Mean_Origin_Error_mm', ...
    'Std_Origin_Error_mm', ...
    'P95_Origin_Error_mm', ...
    'Mean_Rotation_Error_deg', ...
    'Std_Rotation_Error_deg', ...
    'P95_Rotation_Error_deg'});

disp(' ');
disp('================ Random Blind Scan Group Summary ================');
disp(group_summary_table);

%% ============================================================
%  Threshold-based acceptance analysis
% =============================================================

threshold_list = [3, 6, 10];

num_thresholds = length(threshold_list);

pass_count = zeros(num_thresholds, 1);
pass_percent = zeros(num_thresholds, 1);

pass_mean_origin = nan(num_thresholds, 1);
pass_p95_origin = nan(num_thresholds, 1);

pass_mean_rot = nan(num_thresholds, 1);
pass_p95_rot = nan(num_thresholds, 1);

pass_mean_cond = nan(num_thresholds, 1);
pass_mean_inlier = nan(num_thresholds, 1);

for t = 1:num_thresholds

    threshold = threshold_list(t);

    idx = condition_est < threshold;

    pass_count(t) = sum(idx);
    pass_percent(t) = 100 * pass_count(t) / num_valid;

    if pass_count(t) > 0
        pass_mean_cond(t) = mean(condition_est(idx));
        pass_mean_inlier(t) = mean(mean_inlier_ratio(idx));

        pass_mean_origin(t) = mean(origin_error(idx));
        pass_p95_origin(t) = percentile_simple(origin_error(idx), 95);

        pass_mean_rot(t) = mean(rotation_error(idx));
        pass_p95_rot(t) = percentile_simple(rotation_error(idx), 95);
    end
end

threshold_summary_table = table( ...
    threshold_list(:), ...
    pass_count(:), ...
    pass_percent(:), ...
    pass_mean_cond(:), ...
    pass_mean_inlier(:), ...
    pass_mean_origin(:), ...
    pass_p95_origin(:), ...
    pass_mean_rot(:), ...
    pass_p95_rot(:), ...
    'VariableNames', { ...
    'Condition_Threshold', ...
    'Accepted_Count', ...
    'Accepted_Percent', ...
    'Mean_Condition_Number', ...
    'Mean_RANSAC_Inlier_Ratio', ...
    'Mean_Origin_Error_mm', ...
    'P95_Origin_Error_mm', ...
    'Mean_Rotation_Error_deg', ...
    'P95_Rotation_Error_deg'});

disp(' ');
disp('================ Condition Threshold Acceptance Summary ================');
disp(threshold_summary_table);

%% ============================================================
%  Overall summary
% =============================================================

fprintf('\n================ Overall Random Blind Scan Result ================\n');
fprintf('Mean estimated condition number: %.4f\n', mean(condition_est));
fprintf('Median estimated condition number: %.4f\n', median(condition_est));
fprintf('Mean origin error: %.4f mm\n', mean(origin_error));
fprintf('95th percentile origin error: %.4f mm\n', percentile_simple(origin_error, 95));
fprintf('Mean rotation error: %.6f deg\n', mean(rotation_error));
fprintf('95th percentile rotation error: %.6f deg\n', percentile_simple(rotation_error, 95));

%% ============================================================
%  Plots
% =============================================================

figure;
histogram(condition_est, 40);
grid on;
xlabel('Estimated condition number');
ylabel('Count');
title('Distribution of condition number in random blind-scan trials');

figure;
semilogx(condition_est, origin_error, 'o');
grid on;
xlabel('Estimated condition number');
ylabel('Origin error / mm');
title('Origin error versus estimated condition number');

figure;
semilogx(condition_est, rotation_error, 'o');
grid on;
xlabel('Estimated condition number');
ylabel('Rotation error / degree');
title('Rotation error versus estimated condition number');

figure;
boxplot(origin_error, discretize(condition_est, [0 3 6 10 inf]), ...
    'Labels', group_names);
grid on;
ylabel('Origin error / mm');
title('Origin error grouped by condition number');
xtickangle(20);

figure;
boxplot(rotation_error, discretize(condition_est, [0 3 6 10 inf]), ...
    'Labels', group_names);
grid on;
ylabel('Rotation error / degree');
title('Rotation error grouped by condition number');
xtickangle(20);

figure;
bar(sample_percent);
grid on;
set(gca, 'XTickLabel', group_names);
xtickangle(20);
ylabel('Sample percentage / %');
title('Percentage of random board configurations in each condition group');

%% ============================================================
%  Local functions
% =============================================================

function v = random_unit_vector()
    v = randn(3,1);
    v = v / norm(v);
end

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

    e2_temp = n2 - dot(e1, n2) * e1;

    if norm(e2_temp) < 1e-10
        % Fallback for extremely degenerate cases.
        % This case should normally be rejected by condition-number checking.
        tmp = [1; 0; 0];
        if abs(dot(tmp, e1)) > 0.9
            tmp = [0; 1; 0];
        end
        e2_temp = tmp - dot(tmp, e1) * e1;
    end

    e2 = e2_temp / norm(e2_temp);

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

    if n == 0
        p = NaN;
        return;
    end

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