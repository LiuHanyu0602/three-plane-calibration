clc;
clear;
close all;

rng(10);

%% ============================================================
%  Refined Sequential RANSAC Blind Segmentation
%  for Three-Plane LiDAR Calibration
%
%  Goal:
%  1. Generate three randomly placed planar boards.
%  2. Mix all points into one blind point cloud.
%  3. Use Sequential RANSAC to extract three initial planes.
%  4. Refine the three planes by nearest-plane reassignment.
%  5. Compute the condition number of the plane-normal matrix.
%  6. Estimate the board-defined coordinate frame.
%  7. Evaluate origin and rotation calibration accuracy.
%
%  This version improves the previous blind segmentation version by adding
%  a multi-plane refinement stage after initial Sequential RANSAC.
% =============================================================

%% ============================================================
%  Basic simulation settings
% =============================================================

num_trials = 300;

num_points_per_board = 1200;
board_size_1 = 1.0;
board_size_2 = 0.8;

noise_sigma = 0.002;        % 2 mm Gaussian point noise
outlier_ratio = 0.08;       % 8% outliers on each board
outlier_magnitude = 0.08;   % 80 mm outlier displacement along board normal

% Optional background clutter.
% Start with 0. Later you can increase this to 300, 600, etc.
num_background_points = 0;
background_box_size = 3.0;

% Initial Sequential RANSAC parameters
ransac_iter = 800;
ransac_threshold = 0.008;        % 8 mm initial RANSAC threshold
min_plane_inlier_count = 500;

% Multi-plane refinement parameters
refine_threshold = 0.004;        % 4 mm stricter reassignment threshold
refine_iter = 5;                 % number of refinement iterations
min_refine_inlier_count = 300;

%% ============================================================
%  LiDAR pose relative to global simulation frame
% =============================================================

yaw   = deg2rad(35);
pitch = deg2rad(-10);
roll  = deg2rad(8);

R_WL = eulzyx(yaw, pitch, roll);
t_WL = [1.20; -0.60; 0.80];

%% ============================================================
%  Storage
% =============================================================

success_flag = false(num_trials, 1);

condition_true = nan(num_trials, 1);
condition_est = nan(num_trials, 1);

origin_error = nan(num_trials, 1);
rotation_error = nan(num_trials, 1);

mean_plane_purity = nan(num_trials, 1);
min_plane_purity = nan(num_trials, 1);

mean_extracted_inlier_ratio = nan(num_trials, 1);
extracted_inlier_counts = nan(num_trials, 3);

%% ============================================================
%  Main simulation loop
% =============================================================

for trial = 1:num_trials

    %% --------------------------------------------------------
    % 1. Randomly generate three plane normals
    %% --------------------------------------------------------

    nW = zeros(3,3);

    for i = 1:3
        nW(:,i) = random_unit_vector();
    end

    condition_true(trial) = cond(nW');

    % Random three-plane intersection point.
    % This is the origin of the board-defined world frame.
    O_W = [0.30; -0.20; 0.40] + 0.30 * randn(3,1);

    dW = zeros(3,1);
    for i = 1:3
        dW(i) = -nW(:,i)' * O_W;
    end

    %% --------------------------------------------------------
    % 2. Ground-truth board-defined frame
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
    % 4. Generate three board point clouds and mix them
    %% --------------------------------------------------------

    cloud_L = [];
    cloud_label = [];

    for i = 1:3

        % The finite board patch is not necessarily centered at the
        % three-plane intersection point.
        tangent_shift = [0.50 * randn; 0.50 * randn];

        P_W = sample_points_on_plane( ...
            nW(:,i), O_W, tangent_shift, ...
            num_points_per_board, board_size_1, board_size_2);

        % Transform world points into LiDAR frame
        P_L = (R_WL' * (P_W' - t_WL))';

        % Add Gaussian noise
        P_L = P_L + noise_sigma * randn(size(P_L));

        % Add board-related outliers along the true plane normal
        num_outliers = round(outlier_ratio * num_points_per_board);

        if num_outliers > 0
            outlier_idx = randperm(num_points_per_board, num_outliers);

            for k = 1:num_outliers
                idx = outlier_idx(k);
                displacement = outlier_magnitude * (2 * rand - 1);
                P_L(idx,:) = P_L(idx,:) + displacement * nL_true(:,i)';
            end
        end

        cloud_L = [cloud_L; P_L];
        cloud_label = [cloud_label; i * ones(num_points_per_board, 1)];
    end

    % Optional background clutter
    if num_background_points > 0
        bg = background_box_size * (rand(num_background_points, 3) - 0.5);
        cloud_L = [cloud_L; bg];
        cloud_label = [cloud_label; zeros(num_background_points, 1)];
    end

    % Shuffle point order to make the input truly blind
    perm_cloud = randperm(size(cloud_L,1));
    cloud_L = cloud_L(perm_cloud,:);
    cloud_label = cloud_label(perm_cloud);

    %% --------------------------------------------------------
    % 5. Sequential RANSAC initial plane extraction
    %% --------------------------------------------------------

    remaining_points = cloud_L;
    remaining_labels = cloud_label;
    remaining_global_indices = (1:size(cloud_L,1))';

    n_extract = zeros(3,3);
    d_extract = zeros(3,1);
    extracted_labels = cell(3,1);
    extracted_global_inlier_indices = cell(3,1);

    valid_extraction = true;

    for k = 1:3

        [n_r, d_r, inlier_mask] = ransac_plane_fit( ...
            remaining_points, ransac_iter, ransac_threshold);

        inlier_count = sum(inlier_mask);

        if inlier_count < min_plane_inlier_count
            valid_extraction = false;
            break;
        end

        n_extract(:,k) = n_r;
        d_extract(k) = d_r;

        extracted_global_inlier_indices{k} = remaining_global_indices(inlier_mask);
        extracted_labels{k} = remaining_labels(inlier_mask);

        extracted_inlier_counts(trial,k) = inlier_count;

        % Remove extracted plane inliers before extracting next plane
        remaining_points = remaining_points(~inlier_mask,:);
        remaining_labels = remaining_labels(~inlier_mask,:);
        remaining_global_indices = remaining_global_indices(~inlier_mask);
    end

    if ~valid_extraction
        continue;
    end

    %% --------------------------------------------------------
    % 5.5 Multi-plane refinement by nearest-plane reassignment
    %% --------------------------------------------------------

    [n_extract, d_extract, refined_labels, refine_valid] = ...
        refine_three_planes_by_reassignment( ...
            cloud_L, cloud_label, ...
            n_extract, d_extract, ...
            refine_threshold, refine_iter, ...
            min_refine_inlier_count);

    if ~refine_valid
        continue;
    end

    % Replace extracted labels with refined labels
    for k = 1:3
        extracted_labels{k} = refined_labels{k};
        extracted_inlier_counts(trial,k) = length(refined_labels{k});
    end

    %% --------------------------------------------------------
    % 6. Match extracted planes to true planes for evaluation
    %
    % In real experiments, plane order is arbitrary. In simulation, we match
    % extracted planes to ground-truth planes by maximum normal similarity.
    %% --------------------------------------------------------

    [nL_est, dL_est, match_perm] = match_planes_to_truth( ...
        n_extract, d_extract, nL_true);

    % Reorder extracted labels according to matched order
    matched_labels = cell(3,1);

    for i = 1:3
        matched_labels{i} = extracted_labels{match_perm(i)};
    end

    %% --------------------------------------------------------
    % 7. Plane segmentation purity
    %% --------------------------------------------------------

    purity_each = zeros(3,1);

    for i = 1:3
        labels_i = matched_labels{i};

        if isempty(labels_i)
            purity_each(i) = 0;
        else
            count1 = sum(labels_i == 1);
            count2 = sum(labels_i == 2);
            count3 = sum(labels_i == 3);
            count0 = sum(labels_i == 0);

            purity_each(i) = max([count1, count2, count3, count0]) / length(labels_i);
        end
    end

    mean_plane_purity(trial) = mean(purity_each);
    min_plane_purity(trial) = min(purity_each);

    mean_extracted_inlier_ratio(trial) = ...
        mean(extracted_inlier_counts(trial,:) / size(cloud_L,1));

    %% --------------------------------------------------------
    % 8. Condition number and calibration
    %% --------------------------------------------------------

    condition_est(trial) = cond(nL_est');

    [O_L_est, R_LB_est] = frame_from_three_planes(nL_est, dL_est);

    origin_error(trial) = norm(O_L_est - O_L_true) * 1000;
    rotation_error(trial) = rotation_error_deg(R_LB_est, R_LB_true);

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

mean_plane_purity = mean_plane_purity(valid_idx);
min_plane_purity = min_plane_purity(valid_idx);

mean_extracted_inlier_ratio = mean_extracted_inlier_ratio(valid_idx);
extracted_inlier_counts = extracted_inlier_counts(valid_idx,:);

num_valid = length(origin_error);
success_rate = 100 * num_valid / num_trials;

fprintf('\nValid refined blind-segmentation trials: %d / %d\n', num_valid, num_trials);
fprintf('Refined blind segmentation success rate: %.2f %%\n', success_rate);

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

mean_purity = nan(num_groups, 1);
mean_min_purity = nan(num_groups, 1);

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

        mean_purity(g) = mean(mean_plane_purity(idx));
        mean_min_purity(g) = mean(min_plane_purity(idx));
    end
end

group_summary_table = table( ...
    group_names(:), ...
    sample_count(:), ...
    sample_percent(:), ...
    mean_cond_est(:), ...
    mean_cond_true(:), ...
    mean_purity(:), ...
    mean_min_purity(:), ...
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
    'Mean_Segmentation_Purity', ...
    'Mean_Min_Plane_Purity', ...
    'Mean_Origin_Error_mm', ...
    'Std_Origin_Error_mm', ...
    'P95_Origin_Error_mm', ...
    'Mean_Rotation_Error_deg', ...
    'Std_Rotation_Error_deg', ...
    'P95_Rotation_Error_deg'});

disp(' ');
disp('================ Refined Sequential RANSAC Blind Segmentation Group Summary ================');
disp(group_summary_table);

%% ============================================================
%  Threshold acceptance analysis
% =============================================================

threshold_list = [3, 6, 10];

num_thresholds = length(threshold_list);

pass_count = zeros(num_thresholds, 1);
pass_percent = zeros(num_thresholds, 1);

pass_mean_cond = nan(num_thresholds, 1);
pass_mean_purity = nan(num_thresholds, 1);

pass_mean_origin = nan(num_thresholds, 1);
pass_p95_origin = nan(num_thresholds, 1);

pass_mean_rot = nan(num_thresholds, 1);
pass_p95_rot = nan(num_thresholds, 1);

for t = 1:num_thresholds

    threshold = threshold_list(t);

    idx = condition_est < threshold;

    pass_count(t) = sum(idx);
    pass_percent(t) = 100 * pass_count(t) / num_valid;

    if pass_count(t) > 0
        pass_mean_cond(t) = mean(condition_est(idx));
        pass_mean_purity(t) = mean(mean_plane_purity(idx));

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
    pass_mean_purity(:), ...
    pass_mean_origin(:), ...
    pass_p95_origin(:), ...
    pass_mean_rot(:), ...
    pass_p95_rot(:), ...
    'VariableNames', { ...
    'Condition_Threshold', ...
    'Accepted_Count', ...
    'Accepted_Percent', ...
    'Mean_Condition_Number', ...
    'Mean_Segmentation_Purity', ...
    'Mean_Origin_Error_mm', ...
    'P95_Origin_Error_mm', ...
    'Mean_Rotation_Error_deg', ...
    'P95_Rotation_Error_deg'});

disp(' ');
disp('================ Refined Blind Segmentation Threshold Acceptance Summary ================');
disp(threshold_summary_table);

%% ============================================================
%  Overall result
% =============================================================

fprintf('\n================ Overall Refined Blind Segmentation Result ================\n');
fprintf('Valid trials: %d / %d\n', num_valid, num_trials);
fprintf('Success rate: %.2f %%\n', success_rate);
fprintf('Mean segmentation purity: %.4f\n', mean(mean_plane_purity));
fprintf('Mean minimum plane purity: %.4f\n', mean(min_plane_purity));
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
title('Estimated condition number after refined blind sequential RANSAC');

figure;
semilogx(condition_est, origin_error, 'o');
grid on;
xlabel('Estimated condition number');
ylabel('Origin error / mm');
title('Origin error versus condition number after refined blind segmentation');

figure;
semilogx(condition_est, rotation_error, 'o');
grid on;
xlabel('Estimated condition number');
ylabel('Rotation error / degree');
title('Rotation error versus condition number after refined blind segmentation');

figure;
boxplot(origin_error, discretize(condition_est, [0 3 6 10 inf]), ...
    'Labels', group_names);
grid on;
ylabel('Origin error / mm');
title('Origin error grouped by condition number after refinement');
xtickangle(20);

figure;
boxplot(rotation_error, discretize(condition_est, [0 3 6 10 inf]), ...
    'Labels', group_names);
grid on;
ylabel('Rotation error / degree');
title('Rotation error grouped by condition number after refinement');
xtickangle(20);

figure;
boxplot(mean_plane_purity, discretize(condition_est, [0 3 6 10 inf]), ...
    'Labels', group_names);
grid on;
ylabel('Mean segmentation purity');
title('Segmentation purity grouped by condition number after refinement');
xtickangle(20);

figure;
bar(sample_percent);
grid on;
set(gca, 'XTickLabel', group_names);
xtickangle(20);
ylabel('Sample percentage / %');
title('Percentage of refined blind scans in each condition group');

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

    if N < 3
        return;
    end

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

    % Refit by SVD using inliers
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

function [n_refined, d_refined, refined_labels, valid] = ...
    refine_three_planes_by_reassignment( ...
        cloud, labels, n_init, d_init, ...
        threshold, maxIter, minCount)

    % Multi-plane refinement:
    %
    % Given three initial plane models, repeatedly:
    % 1. Compute each point's distance to each plane.
    % 2. Assign each point to its nearest plane.
    % 3. Keep only points whose nearest-plane distance is below threshold.
    % 4. Refit each plane using SVD.
    %
    % This suppresses cross-plane contamination left by initial Sequential
    % RANSAC extraction.

    n_refined = n_init;
    d_refined = d_init;

    refined_labels = cell(3,1);
    valid = true;

    num_points = size(cloud,1);

    for iter = 1:maxIter

        residuals = zeros(num_points, 3);

        for k = 1:3
            residuals(:,k) = abs(cloud * n_refined(:,k) + d_refined(k));
        end

        [min_res, assigned_plane] = min(residuals, [], 2);

        new_n = zeros(3,3);
        new_d = zeros(3,1);
        new_refined_labels = cell(3,1);

        for k = 1:3

            idx = (assigned_plane == k) & (min_res < threshold);

            if sum(idx) < minCount
                valid = false;
                return;
            end

            Pk = cloud(idx,:);

            [n_k, d_k] = fit_plane_svd(Pk);

            % Keep normal sign consistent with previous iteration
            if dot(n_k, n_refined(:,k)) < 0
                n_k = -n_k;
                d_k = -d_k;
            end

            new_n(:,k) = n_k;
            new_d(k) = d_k;
            new_refined_labels{k} = labels(idx);
        end

        n_refined = new_n;
        d_refined = new_d;
        refined_labels = new_refined_labels;
    end
end

function [n_sorted, d_sorted, best_perm] = match_planes_to_truth(n_est, d_est, n_true)
    % Match unordered extracted planes to ground-truth planes by maximum
    % absolute normal similarity.
    %
    % best_perm(i) gives the extracted plane index matched to true plane i.

    P = perms(1:3);

    best_score = -inf;
    best_perm = P(1,:);

    for k = 1:size(P,1)
        perm = P(k,:);

        score = 0;
        for i = 1:3
            score = score + abs(dot(n_est(:,perm(i)), n_true(:,i)));
        end

        if score > best_score
            best_score = score;
            best_perm = perm;
        end
    end

    n_sorted = zeros(3,3);
    d_sorted = zeros(3,1);

    for i = 1:3
        idx = best_perm(i);

        n_i = n_est(:,idx);
        d_i = d_est(idx);

        % Align sign for simulation evaluation
        if dot(n_i, n_true(:,i)) < 0
            n_i = -n_i;
            d_i = -d_i;
        end

        n_sorted(:,i) = n_i;
        d_sorted(i) = d_i;
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