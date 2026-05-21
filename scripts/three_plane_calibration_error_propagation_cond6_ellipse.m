clc;
clear;
close all;

rng(10);

%% ============================================================
%  Refined Sequential RANSAC Blind Segmentation
%  for Three-Plane LiDAR Calibration
%
%  Updated version:
%  1. Keeps the original refined Sequential RANSAC + nearest-plane
%     reassignment pipeline.
%  2. Adds Monte Carlo origin-position error vector analysis:
%        Delta O = O_est - O_true = [dx, dy, dz]^T.
%  3. Reports mean/std/P95 of dx, dy, dz and norm error.
%  4. Computes covariance matrix of origin-position error.
%  5. Draws 95% confidence ellipses in dx-dy, dx-dz, dy-dz projections.
%  6. Adds ablation groups to separate:
%        A: no noise, no outliers
%        B: point-cloud noise only
%        C: outliers only
%        D: noise + outliers
%  7. Adds a post-hoc bias-correction check.
%  8. Adds accepted-sample error ellipses for kappa(N)<6,
%     so the error distribution after geometric observability filtering
%     can be compared with the all-sample distribution.
%
%  Important terminology:
%  "Origin error" is renamed as "origin-position / translation error".
%  It is NOT a single ranging error. It is:
%        || O_est - O_true ||_2
%  after point-cloud noise, plane extraction, plane fitting,
%  and three-plane intersection/calibration.
% =============================================================

%% ============================================================
%  Basic simulation settings
% =============================================================

cfg = struct();

cfg.num_trials = 300;

cfg.num_points_per_board = 1200;
cfg.board_size_1 = 1.0;
cfg.board_size_2 = 0.8;

% Noise/outlier settings will be overwritten by experiment groups below.
cfg.noise_sigma = 0.002;        % 2 mm Gaussian point noise
cfg.outlier_ratio = 0.08;       % 8% outliers on each board
cfg.outlier_magnitude = 0.08;   % 80 mm outlier displacement along board normal

% Optional background clutter.
% Start with 0. Later you can increase this to 300, 600, etc.
cfg.num_background_points = 0;
cfg.background_box_size = 3.0;

% Initial Sequential RANSAC parameters
cfg.ransac_iter = 800;
cfg.ransac_threshold = 0.008;        % 8 mm initial RANSAC threshold
cfg.min_plane_inlier_count = 500;

% Multi-plane refinement parameters
cfg.refine_threshold = 0.004;        % 4 mm stricter reassignment threshold
cfg.refine_iter = 5;                 % number of refinement iterations
cfg.min_refine_inlier_count = 300;

% Fixed LiDAR pose relative to global simulation frame
cfg.yaw   = deg2rad(35);
cfg.pitch = deg2rad(-10);
cfg.roll  = deg2rad(8);

cfg.R_WL = eulzyx(cfg.yaw, cfg.pitch, cfg.roll);
cfg.t_WL = [1.20; -0.60; 0.80];

%% ============================================================
%  Ablation experiment settings
% =============================================================

experiment_groups = struct( ...
    'Name', { ...
        'A_NoNoise_NoOutlier', ...
        'B_NoiseOnly_2mm', ...
        'C_OutlierOnly_8percent', ...
        'D_Noise2mm_Outlier8percent'}, ...
    'NoiseSigma_m', {0.000, 0.002, 0.000, 0.002}, ...
    'OutlierRatio', {0.00, 0.00, 0.08, 0.08});

% The main group is the realistic case used for detailed plots and
% condition-number analysis.
main_group_index = 4;

num_experiment_groups = numel(experiment_groups);
all_group_data = cell(num_experiment_groups, 1);
all_ablation_summary = table();

%% ============================================================
%  Run ablation groups
% =============================================================

for g = 1:num_experiment_groups

    cfg_g = cfg;
    cfg_g.noise_sigma = experiment_groups(g).NoiseSigma_m;
    cfg_g.outlier_ratio = experiment_groups(g).OutlierRatio;
    cfg_g.group_name = experiment_groups(g).Name;

    fprintf('\n\n============================================================\n');
    fprintf('Running experiment group %d / %d: %s\n', ...
        g, num_experiment_groups, cfg_g.group_name);
    fprintf('Point-cloud noise sigma: %.3f mm\n', 1000 * cfg_g.noise_sigma);
    fprintf('Outlier ratio: %.2f %%\n', 100 * cfg_g.outlier_ratio);
    fprintf('============================================================\n');

    % Use a different but repeatable seed for each ablation group.
    rng(10 + 1000 * g);

    data_g = run_three_plane_monte_carlo(cfg_g);
    all_group_data{g} = data_g;

    summary_g = summarize_ablation_group(data_g, cfg_g.group_name, ...
        1000 * cfg_g.noise_sigma, 100 * cfg_g.outlier_ratio);

    all_ablation_summary = [all_ablation_summary; summary_g];

    print_origin_error_vector_statistics(data_g, cfg_g.group_name);
end

disp(' ');
disp('================ Ablation Summary: Error Source Separation ================');
disp(all_ablation_summary);

%% ============================================================
%  Detailed analysis for the main group: 2 mm noise + 8% outliers
% =============================================================

main_data = all_group_data{main_group_index};
main_name = experiment_groups(main_group_index).Name;

fprintf('\n\n============================================================\n');
fprintf('Detailed analysis for main group: %s\n', main_name);
fprintf('============================================================\n');

%% ------------------------------------------------------------
%  Condition-number group analysis
%% ------------------------------------------------------------

[group_summary_table, threshold_summary_table] = ...
    build_condition_analysis_tables(main_data);

disp(' ');
disp('================ Refined Sequential RANSAC Blind Segmentation Group Summary ================');
disp(group_summary_table);

disp(' ');
disp('================ Refined Blind Segmentation Threshold Acceptance Summary ================');
disp(threshold_summary_table);

%% ------------------------------------------------------------
%  Overall main-group result
%% ------------------------------------------------------------

fprintf('\n================ Overall Refined Blind Segmentation Result: %s ================\n', main_name);
fprintf('Valid trials: %d / %d\n', main_data.num_valid, main_data.num_trials);
fprintf('Success rate: %.2f %%\n', main_data.success_rate);
fprintf('Mean segmentation purity: %.4f\n', mean(main_data.mean_plane_purity));
fprintf('Mean minimum plane purity: %.4f\n', mean(main_data.min_plane_purity));
fprintf('Mean estimated condition number: %.4f\n', mean(main_data.condition_est));
fprintf('Median estimated condition number: %.4f\n', median(main_data.condition_est));
fprintf('Mean origin-position / translation error: %.4f mm\n', ...
    mean(main_data.origin_position_error_mm));
fprintf('95th percentile origin-position / translation error: %.4f mm\n', ...
    percentile_simple(main_data.origin_position_error_mm, 95));
fprintf('Mean rotation error: %.6f deg\n', mean(main_data.rotation_error_deg));
fprintf('95th percentile rotation error: %.6f deg\n', ...
    percentile_simple(main_data.rotation_error_deg, 95));

%% ------------------------------------------------------------
%  Post-hoc bias-correction check for origin-position error
%% ------------------------------------------------------------

bias_correction_table = compute_bias_correction_check(main_data);

disp(' ');
disp('================ Post-hoc Bias Correction Check ================');
disp(bias_correction_table);

fprintf(['\nNote: This bias correction is only an analysis tool.\n', ...
         'If the mean error vector is stable across configurations, it can be used as a calibration compensation term.\n', ...
         'If it changes with plane geometry/noise/outliers, the better solution is stronger constraints and weighted fitting.\n']);

%% ============================================================
%  Plots for main group
% =============================================================

plot_main_group_results(main_data, main_name);

%% ============================================================
%  Save numerical results to .mat file
% =============================================================

save('three_plane_calibration_error_propagation_results.mat', ...
    'cfg', 'experiment_groups', 'all_group_data', ...
    'all_ablation_summary', 'group_summary_table', ...
    'threshold_summary_table', 'bias_correction_table');

fprintf('\nSaved results to three_plane_calibration_error_propagation_results.mat\n');

%% ============================================================
%  Local functions
% =============================================================

function data = run_three_plane_monte_carlo(cfg)

    num_trials = cfg.num_trials;

    %% --------------------------------------------------------
    %  Storage
    %% --------------------------------------------------------

    success_flag = false(num_trials, 1);

    condition_true = nan(num_trials, 1);
    condition_est = nan(num_trials, 1);

    origin_position_error_mm = nan(num_trials, 1);
    origin_error_vec_mm = nan(num_trials, 3);

    rotation_error_deg_arr = nan(num_trials, 1);

    mean_plane_purity = nan(num_trials, 1);
    min_plane_purity = nan(num_trials, 1);

    mean_extracted_inlier_ratio = nan(num_trials, 1);
    extracted_inlier_counts = nan(num_trials, 3);

    min_plane_angle_true_deg = nan(num_trials, 1);
    min_plane_angle_est_deg = nan(num_trials, 1);

    %% --------------------------------------------------------
    %  Main simulation loop
    %% --------------------------------------------------------

    for trial = 1:num_trials

        %% ----------------------------------------------------
        % 1. Randomly generate three plane normals
        %% ----------------------------------------------------

        nW = zeros(3,3);

        for i = 1:3
            nW(:,i) = random_unit_vector();
        end

        condition_true(trial) = cond(nW');
        min_plane_angle_true_deg(trial) = min_pairwise_normal_angle_deg(nW);

        % Random three-plane intersection point.
        % This is the origin of the board-defined world frame.
        O_W = [0.30; -0.20; 0.40] + 0.30 * randn(3,1);

        dW = zeros(3,1);
        for i = 1:3
            dW(i) = -nW(:,i)' * O_W;
        end

        %% ----------------------------------------------------
        % 2. Ground-truth board-defined frame
        %% ----------------------------------------------------

        [O_W_check, R_WB] = frame_from_three_planes(nW, dW);

        O_L_true = cfg.R_WL' * (O_W_check - cfg.t_WL);
        R_LB_true = cfg.R_WL' * R_WB;

        %% ----------------------------------------------------
        % 3. True plane parameters in LiDAR frame
        %% ----------------------------------------------------

        nL_true = zeros(3,3);
        dL_true = zeros(3,1);

        for i = 1:3
            nL_true(:,i) = cfg.R_WL' * nW(:,i);
            dL_true(i) = nW(:,i)' * cfg.t_WL + dW(i);
        end

        %% ----------------------------------------------------
        % 4. Generate three board point clouds and mix them
        %% ----------------------------------------------------

        cloud_L = [];
        cloud_label = [];

        for i = 1:3

            % The finite board patch is not necessarily centered at the
            % three-plane intersection point.
            tangent_shift = [0.50 * randn; 0.50 * randn];

            P_W = sample_points_on_plane( ...
                nW(:,i), O_W, tangent_shift, ...
                cfg.num_points_per_board, cfg.board_size_1, cfg.board_size_2);

            % Transform world points into LiDAR frame
            P_L = (cfg.R_WL' * (P_W' - cfg.t_WL))';

            % Add Gaussian point-cloud / ranging noise
            if cfg.noise_sigma > 0
                P_L = P_L + cfg.noise_sigma * randn(size(P_L));
            end

            % Add board-related outliers along the true plane normal
            num_outliers = round(cfg.outlier_ratio * cfg.num_points_per_board);

            if num_outliers > 0
                outlier_idx = randperm(cfg.num_points_per_board, num_outliers);

                for k = 1:num_outliers
                    idx = outlier_idx(k);
                    displacement = cfg.outlier_magnitude * (2 * rand - 1);
                    P_L(idx,:) = P_L(idx,:) + displacement * nL_true(:,i)';
                end
            end

            cloud_L = [cloud_L; P_L];
            cloud_label = [cloud_label; i * ones(cfg.num_points_per_board, 1)];
        end

        % Optional background clutter
        if cfg.num_background_points > 0
            bg = cfg.background_box_size * (rand(cfg.num_background_points, 3) - 0.5);
            cloud_L = [cloud_L; bg];
            cloud_label = [cloud_label; zeros(cfg.num_background_points, 1)];
        end

        % Shuffle point order to make the input truly blind
        perm_cloud = randperm(size(cloud_L,1));
        cloud_L = cloud_L(perm_cloud,:);
        cloud_label = cloud_label(perm_cloud);

        %% ----------------------------------------------------
        % 5. Sequential RANSAC initial plane extraction
        %% ----------------------------------------------------

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
                remaining_points, cfg.ransac_iter, cfg.ransac_threshold);

            inlier_count = sum(inlier_mask);

            if inlier_count < cfg.min_plane_inlier_count
                valid_extraction = false;
                break;
            end

            n_extract(:,k) = n_r;
            d_extract(k) = d_r;

            extracted_global_inlier_indices{k} = remaining_global_indices(inlier_mask); %#ok<NASGU>
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

        %% ----------------------------------------------------
        % 5.5 Multi-plane refinement by nearest-plane reassignment
        %% ----------------------------------------------------

        [n_extract, d_extract, refined_labels, refine_valid] = ...
            refine_three_planes_by_reassignment( ...
                cloud_L, cloud_label, ...
                n_extract, d_extract, ...
                cfg.refine_threshold, cfg.refine_iter, ...
                cfg.min_refine_inlier_count);

        if ~refine_valid
            continue;
        end

        % Replace extracted labels with refined labels
        for k = 1:3
            extracted_labels{k} = refined_labels{k};
            extracted_inlier_counts(trial,k) = length(refined_labels{k});
        end

        %% ----------------------------------------------------
        % 6. Match extracted planes to true planes for evaluation
        %
        % In real experiments, plane order is arbitrary. In simulation, we match
        % extracted planes to ground-truth planes by maximum normal similarity.
        %% ----------------------------------------------------

        [nL_est, dL_est, match_perm] = match_planes_to_truth( ...
            n_extract, d_extract, nL_true);

        % Reorder extracted labels according to matched order
        matched_labels = cell(3,1);

        for i = 1:3
            matched_labels{i} = extracted_labels{match_perm(i)};
        end

        %% ----------------------------------------------------
        % 7. Plane segmentation purity
        %% ----------------------------------------------------

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

        %% ----------------------------------------------------
        % 8. Condition number and calibration
        %% ----------------------------------------------------

        condition_est(trial) = cond(nL_est');
        min_plane_angle_est_deg(trial) = min_pairwise_normal_angle_deg(nL_est);

        [O_L_est, R_LB_est] = frame_from_three_planes(nL_est, dL_est);

        % Error vector in mm:
        % Delta O = O_est - O_true.
        % This is the quantity needed for covariance/ellipse analysis.
        delta_origin_mm = 1000 * (O_L_est(:) - O_L_true(:));

        origin_error_vec_mm(trial,:) = delta_origin_mm(:)';
        origin_position_error_mm(trial) = norm(delta_origin_mm);

        rotation_error_deg_arr(trial) = rotation_error_deg(R_LB_est, R_LB_true);

        success_flag(trial) = true;

        if mod(trial, 50) == 0
            fprintf('Finished trial %d / %d\n', trial, num_trials);
        end
    end

    %% --------------------------------------------------------
    %  Remove invalid trials
    %% --------------------------------------------------------

    valid_idx = success_flag & ...
                isfinite(origin_position_error_mm) & ...
                isfinite(rotation_error_deg_arr) & ...
                isfinite(condition_est);

    data = struct();

    data.num_trials = num_trials;
    data.valid_idx = valid_idx;
    data.num_valid = sum(valid_idx);
    data.success_rate = 100 * data.num_valid / num_trials;

    data.condition_true = condition_true(valid_idx);
    data.condition_est = condition_est(valid_idx);

    data.origin_position_error_mm = origin_position_error_mm(valid_idx);
    data.origin_error_vec_mm = origin_error_vec_mm(valid_idx,:);

    data.rotation_error_deg = rotation_error_deg_arr(valid_idx);

    data.mean_plane_purity = mean_plane_purity(valid_idx);
    data.min_plane_purity = min_plane_purity(valid_idx);

    data.mean_extracted_inlier_ratio = mean_extracted_inlier_ratio(valid_idx);
    data.extracted_inlier_counts = extracted_inlier_counts(valid_idx,:);

    data.min_plane_angle_true_deg = min_plane_angle_true_deg(valid_idx);
    data.min_plane_angle_est_deg = min_plane_angle_est_deg(valid_idx);

    fprintf('\nValid refined blind-segmentation trials: %d / %d\n', ...
        data.num_valid, num_trials);
    fprintf('Refined blind segmentation success rate: %.2f %%\n', ...
        data.success_rate);
end

function summary_table = summarize_ablation_group(data, group_name, noise_sigma_mm, outlier_ratio_percent)

    E = data.origin_error_vec_mm;

    if isempty(E)
        summary_table = table( ...
            string(group_name), noise_sigma_mm, outlier_ratio_percent, ...
            0, 0, ...
            NaN, NaN, NaN, NaN, NaN, NaN, ...
            NaN, NaN, NaN, NaN, ...
            'VariableNames', { ...
            'Experiment_Group', ...
            'Noise_sigma_mm', ...
            'Outlier_Ratio_percent', ...
            'Valid_Count', ...
            'Success_Rate_percent', ...
            'Mean_dx_mm', 'Mean_dy_mm', 'Mean_dz_mm', ...
            'Std_dx_mm', 'Std_dy_mm', 'Std_dz_mm', ...
            'Mean_Origin_Position_Error_mm', ...
            'P95_Origin_Position_Error_mm', ...
            'Mean_Rotation_Error_deg', ...
            'P95_Rotation_Error_deg'});
        return;
    end

    dx = E(:,1);
    dy = E(:,2);
    dz = E(:,3);

    summary_table = table( ...
        string(group_name), ...
        noise_sigma_mm, ...
        outlier_ratio_percent, ...
        data.num_valid, ...
        data.success_rate, ...
        mean(dx), mean(dy), mean(dz), ...
        std(dx), std(dy), std(dz), ...
        mean(data.origin_position_error_mm), ...
        percentile_simple(data.origin_position_error_mm, 95), ...
        mean(data.rotation_error_deg), ...
        percentile_simple(data.rotation_error_deg, 95), ...
        'VariableNames', { ...
        'Experiment_Group', ...
        'Noise_sigma_mm', ...
        'Outlier_Ratio_percent', ...
        'Valid_Count', ...
        'Success_Rate_percent', ...
        'Mean_dx_mm', 'Mean_dy_mm', 'Mean_dz_mm', ...
        'Std_dx_mm', 'Std_dy_mm', 'Std_dz_mm', ...
        'Mean_Origin_Position_Error_mm', ...
        'P95_Origin_Position_Error_mm', ...
        'Mean_Rotation_Error_deg', ...
        'P95_Rotation_Error_deg'});
end

function print_origin_error_vector_statistics(data, group_name)

    fprintf('\n================ Origin Error Vector Statistics: %s ================\n', group_name);

    if data.num_valid == 0
        fprintf('No valid trials.\n');
        return;
    end

    E = data.origin_error_vec_mm;
    dx = E(:,1);
    dy = E(:,2);
    dz = E(:,3);

    mu_E = mean(E, 1);
    Sigma_E = cov(E);

    fprintf('Mean dx = %.6f mm\n', mu_E(1));
    fprintf('Mean dy = %.6f mm\n', mu_E(2));
    fprintf('Mean dz = %.6f mm\n', mu_E(3));

    fprintf('Std  dx = %.6f mm\n', std(dx));
    fprintf('Std  dy = %.6f mm\n', std(dy));
    fprintf('Std  dz = %.6f mm\n', std(dz));

    fprintf('\nMean norm origin-position error = %.6f mm\n', ...
        mean(data.origin_position_error_mm));
    fprintf('Std  norm origin-position error = %.6f mm\n', ...
        std(data.origin_position_error_mm));
    fprintf('P95  norm origin-position error = %.6f mm\n', ...
        percentile_simple(data.origin_position_error_mm, 95));

    fprintf('\nCovariance matrix of origin-position error, unit: mm^2\n');
    disp(Sigma_E);

    [V, D] = eig(Sigma_E);
    eig_values = diag(D);
    [eig_values, order] = sort(eig_values, 'descend');
    V = V(:, order);

    fprintf('Eigenvalues of covariance matrix, unit: mm^2\n');
    disp(eig_values');

    fprintf('Principal directions of origin error covariance:\n');
    disp(V);
end

function [group_summary_table, threshold_summary_table] = build_condition_analysis_tables(data)

    condition_group_names = { ...
        'Good: cond < 3', ...
        'Acceptable: 3 <= cond < 6', ...
        'Risky: 6 <= cond < 10', ...
        'Bad: cond >= 10'};

    group_lower = [0, 3, 6, 10];
    group_upper = [3, 6, 10, inf];

    num_groups = length(condition_group_names);

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

    mean_min_angle = nan(num_groups, 1);

    for g = 1:num_groups

        idx = data.condition_est >= group_lower(g) & ...
              data.condition_est < group_upper(g);

        sample_count(g) = sum(idx);
        sample_percent(g) = 100 * sample_count(g) / data.num_valid;

        if sample_count(g) > 0
            mean_cond_est(g) = mean(data.condition_est(idx));
            mean_cond_true(g) = mean(data.condition_true(idx));

            mean_origin(g) = mean(data.origin_position_error_mm(idx));
            std_origin(g) = std(data.origin_position_error_mm(idx));
            p95_origin(g) = percentile_simple(data.origin_position_error_mm(idx), 95);

            mean_rot(g) = mean(data.rotation_error_deg(idx));
            std_rot(g) = std(data.rotation_error_deg(idx));
            p95_rot(g) = percentile_simple(data.rotation_error_deg(idx), 95);

            mean_purity(g) = mean(data.mean_plane_purity(idx));
            mean_min_purity(g) = mean(data.min_plane_purity(idx));
            mean_min_angle(g) = mean(data.min_plane_angle_est_deg(idx));
        end
    end

    group_summary_table = table( ...
        condition_group_names(:), ...
        sample_count(:), ...
        sample_percent(:), ...
        mean_cond_est(:), ...
        mean_cond_true(:), ...
        mean_min_angle(:), ...
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
        'Mean_Min_Plane_Angle_deg', ...
        'Mean_Segmentation_Purity', ...
        'Mean_Min_Plane_Purity', ...
        'Mean_Origin_Position_Error_mm', ...
        'Std_Origin_Position_Error_mm', ...
        'P95_Origin_Position_Error_mm', ...
        'Mean_Rotation_Error_deg', ...
        'Std_Rotation_Error_deg', ...
        'P95_Rotation_Error_deg'});

    %% --------------------------------------------------------
    %  Threshold acceptance analysis
    %% --------------------------------------------------------

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

        idx = data.condition_est < threshold;

        pass_count(t) = sum(idx);
        pass_percent(t) = 100 * pass_count(t) / data.num_valid;

        if pass_count(t) > 0
            pass_mean_cond(t) = mean(data.condition_est(idx));
            pass_mean_purity(t) = mean(data.mean_plane_purity(idx));

            pass_mean_origin(t) = mean(data.origin_position_error_mm(idx));
            pass_p95_origin(t) = percentile_simple(data.origin_position_error_mm(idx), 95);

            pass_mean_rot(t) = mean(data.rotation_error_deg(idx));
            pass_p95_rot(t) = percentile_simple(data.rotation_error_deg(idx), 95);
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
        'Mean_Origin_Position_Error_mm', ...
        'P95_Origin_Position_Error_mm', ...
        'Mean_Rotation_Error_deg', ...
        'P95_Rotation_Error_deg'});
end

function bias_table = compute_bias_correction_check(data)

    E = data.origin_error_vec_mm;

    if isempty(E)
        bias_table = table();
        return;
    end

    bias_vec = mean(E, 1);
    E_corr = E - bias_vec;

    err_before = sqrt(sum(E.^2, 2));
    err_after = sqrt(sum(E_corr.^2, 2));

    bias_table = table( ...
        ["Before bias correction"; "After subtracting mean bias"], ...
        [mean(err_before); mean(err_after)], ...
        [percentile_simple(err_before, 95); percentile_simple(err_after, 95)], ...
        [bias_vec(1); 0], ...
        [bias_vec(2); 0], ...
        [bias_vec(3); 0], ...
        'VariableNames', { ...
        'Case', ...
        'Mean_Origin_Position_Error_mm', ...
        'P95_Origin_Position_Error_mm', ...
        'Mean_dx_mm', ...
        'Mean_dy_mm', ...
        'Mean_dz_mm'});
end

function plot_main_group_results(data, group_name)

    condition_group_names = { ...
        'Good: cond < 3', ...
        'Acceptable: 3 <= cond < 6', ...
        'Risky: 6 <= cond < 10', ...
        'Bad: cond >= 10'};

    % Escape underscores in group name for MATLAB TeX titles.
    group_title = strrep(group_name, '_', '\_');

    E = data.origin_error_vec_mm;
    dx = E(:,1);
    dy = E(:,2);
    dz = E(:,3);

    %% --------------------------------------------------------
    %  Error ellipses for all valid samples
    %
    %  These figures include all valid Monte Carlo trials. They may contain
    %  a few very large points when the three-plane normal matrix is badly
    %  conditioned. Therefore, they show the full uncertainty distribution
    %  before geometric observability filtering.
    %% --------------------------------------------------------

    figure;
    plot_error_ellipse_2d(dx, dy, ...
        '\Delta x / mm', '\Delta y / mm', ...
        ['All valid samples: 95% origin-position error ellipse, \Delta x-\Delta y, ', group_title]);

    figure;
    plot_error_ellipse_2d(dx, dz, ...
        '\Delta x / mm', '\Delta z / mm', ...
        ['All valid samples: 95% origin-position error ellipse, \Delta x-\Delta z, ', group_title]);

    figure;
    plot_error_ellipse_2d(dy, dz, ...
        '\Delta y / mm', '\Delta z / mm', ...
        ['All valid samples: 95% origin-position error ellipse, \Delta y-\Delta z, ', group_title]);

    %% --------------------------------------------------------
    %  Error ellipses for accepted samples: kappa(N) < 6
    %
    %  These are the key figures for the teacher's question. They show the
    %  origin-position uncertainty after rejecting geometrically ill-
    %  conditioned three-plane configurations.
    %% --------------------------------------------------------

    accepted_threshold = 6;
    idx_accept = data.condition_est < accepted_threshold;
    num_accept = sum(idx_accept);

    fprintf('\n================ Accepted-sample Ellipse Analysis ================\n');
    fprintf('Acceptance condition: estimated condition number kappa(N) < %.2f\n', accepted_threshold);
    fprintf('Accepted samples: %d / %d, %.2f %%\n', ...
        num_accept, data.num_valid, 100 * num_accept / data.num_valid);

    if num_accept >= 3

        E_acc = E(idx_accept, :);
        dx_acc = E_acc(:,1);
        dy_acc = E_acc(:,2);
        dz_acc = E_acc(:,3);

        fprintf('Accepted mean origin-position error: %.6f mm\n', ...
            mean(data.origin_position_error_mm(idx_accept)));
        fprintf('Accepted P95 origin-position error: %.6f mm\n', ...
            percentile_simple(data.origin_position_error_mm(idx_accept), 95));
        fprintf('Accepted mean rotation error: %.6f deg\n', ...
            mean(data.rotation_error_deg(idx_accept)));
        fprintf('Accepted P95 rotation error: %.6f deg\n', ...
            percentile_simple(data.rotation_error_deg(idx_accept), 95));

        Sigma_acc = cov(E_acc);
        fprintf('Accepted origin-position error covariance matrix, unit: mm^2\n');
        disp(Sigma_acc);

        figure;
        plot_error_ellipse_2d(dx_acc, dy_acc, ...
            '\Delta x / mm', '\Delta y / mm', ...
            ['Accepted samples only, \kappa(N)<6: 95% error ellipse, \Delta x-\Delta y, ', group_title]);

        figure;
        plot_error_ellipse_2d(dx_acc, dz_acc, ...
            '\Delta x / mm', '\Delta z / mm', ...
            ['Accepted samples only, \kappa(N)<6: 95% error ellipse, \Delta x-\Delta z, ', group_title]);

        figure;
        plot_error_ellipse_2d(dy_acc, dz_acc, ...
            '\Delta y / mm', '\Delta z / mm', ...
            ['Accepted samples only, \kappa(N)<6: 95% error ellipse, \Delta y-\Delta z, ', group_title]);

    else
        fprintf('Too few accepted samples to draw a reliable 95%% ellipse.\n');
    end

    %% --------------------------------------------------------
    %  Optional comparison ellipses: all samples versus accepted samples
    %
    %  These overlay figures make the effect of kappa(N)<6 visually clear.
    %% --------------------------------------------------------

    if num_accept >= 3
        figure;
        plot_error_ellipse_comparison_2d(dx, dy, dx_acc, dy_acc, ...
            '\Delta x / mm', '\Delta y / mm', ...
            ['All samples vs accepted \kappa(N)<6: \Delta x-\Delta y, ', group_title]);

        figure;
        plot_error_ellipse_comparison_2d(dx, dz, dx_acc, dz_acc, ...
            '\Delta x / mm', '\Delta z / mm', ...
            ['All samples vs accepted \kappa(N)<6: \Delta x-\Delta z, ', group_title]);

        figure;
        plot_error_ellipse_comparison_2d(dy, dz, dy_acc, dz_acc, ...
            '\Delta y / mm', '\Delta z / mm', ...
            ['All samples vs accepted \kappa(N)<6: \Delta y-\Delta z, ', group_title]);
    end

    %% --------------------------------------------------------
    %  Original diagnostic plots
    %% --------------------------------------------------------

    figure;
    histogram(data.condition_est, 40);
    grid on;
    xlabel('Estimated condition number');
    ylabel('Count');
    title(['Estimated condition number after refined blind sequential RANSAC, ', group_title]);

    figure;
    semilogx(data.condition_est, data.origin_position_error_mm, 'o');
    grid on;
    xlabel('Estimated condition number');
    ylabel('Origin-position / translation error / mm');
    title(['Origin-position error versus condition number, ', group_title]);

    figure;
    semilogx(data.condition_est, data.rotation_error_deg, 'o');
    grid on;
    xlabel('Estimated condition number');
    ylabel('Rotation error / degree');
    title(['Rotation error versus condition number, ', group_title]);

    figure;
    semilogx(data.min_plane_angle_est_deg, data.origin_position_error_mm, 'o');
    grid on;
    xlabel('Minimum pairwise plane-normal angle / degree');
    ylabel('Origin-position / translation error / mm');
    title(['Origin-position error versus minimum plane angle, ', group_title]);

    figure;
    boxplot(data.origin_position_error_mm, discretize(data.condition_est, [0 3 6 10 inf]), ...
        'Labels', condition_group_names);
    grid on;
    ylabel('Origin-position / translation error / mm');
    title(['Origin-position error grouped by condition number, ', group_title]);
    xtickangle(20);

    figure;
    boxplot(data.rotation_error_deg, discretize(data.condition_est, [0 3 6 10 inf]), ...
        'Labels', condition_group_names);
    grid on;
    ylabel('Rotation error / degree');
    title(['Rotation error grouped by condition number, ', group_title]);
    xtickangle(20);

    figure;
    boxplot(data.mean_plane_purity, discretize(data.condition_est, [0 3 6 10 inf]), ...
        'Labels', condition_group_names);
    grid on;
    ylabel('Mean segmentation purity');
    title(['Segmentation purity grouped by condition number, ', group_title]);
    xtickangle(20);
end

function plot_error_ellipse_comparison_2d(x_all, y_all, x_acc, y_acc, xlabel_name, ylabel_name, title_name)

    % Overlay all-sample and accepted-sample covariance ellipses.
    % This is useful to visually show how the kappa(N)<6 observability
    % constraint suppresses long-tail uncertainty from bad three-plane
    % configurations.

    [ellipse_all, mu_all] = compute_covariance_ellipse_2d(x_all, y_all);
    [ellipse_acc, mu_acc] = compute_covariance_ellipse_2d(x_acc, y_acc);

    scatter(x_all, y_all, 16, 'o');
    hold on;
    scatter(x_acc, y_acc, 20, 'filled');

    plot(ellipse_all(1,:), ellipse_all(2,:), 'LineWidth', 2);
    plot(ellipse_acc(1,:), ellipse_acc(2,:), 'LineWidth', 2);

    plot(mu_all(1), mu_all(2), 'x', 'LineWidth', 2, 'MarkerSize', 10);
    plot(mu_acc(1), mu_acc(2), 'x', 'LineWidth', 2, 'MarkerSize', 10);
    plot(0, 0, 'ko', 'LineWidth', 2, 'MarkerSize', 8);

    grid on;
    axis equal;
    xlabel(xlabel_name);
    ylabel(ylabel_name);
    title(title_name);

    legend('All valid samples', ...
           'Accepted samples, \kappa(N)<6', ...
           'All-sample 95% covariance ellipse', ...
           'Accepted-sample 95% covariance ellipse', ...
           'All-sample mean center', ...
           'Accepted-sample mean center', ...
           'True zero error', ...
           'Location', 'best');
end

function [ellipse, mu] = compute_covariance_ellipse_2d(x, y)

    data = [x(:), y(:)];
    mu = mean(data, 1);
    Sigma = cov(data);

    % 95% covariance ellipse for a 2D Gaussian assumption.
    chi2_val = 5.991;

    [V, D] = eig(Sigma);
    D = max(D, 0);

    theta = linspace(0, 2*pi, 300);
    circle = [cos(theta); sin(theta)];

    ellipse = V * sqrt(D * chi2_val) * circle;
    ellipse(1,:) = ellipse(1,:) + mu(1);
    ellipse(2,:) = ellipse(2,:) + mu(2);
end

function plot_error_ellipse_2d(x, y, xlabel_name, ylabel_name, title_name)

    data = [x(:), y(:)];
    mu = mean(data, 1);
    Sigma = cov(data);

    % 95% confidence ellipse for 2D Gaussian:
    % (x-mu)' Sigma^{-1} (x-mu) <= chi2inv(0.95,2).
    % chi2inv(0.95,2) = 5.991.
    chi2_val = 5.991;

    [V, D] = eig(Sigma);

    % Numerical safety: remove tiny negative eigenvalues caused by roundoff.
    D = max(D, 0);

    theta = linspace(0, 2*pi, 300);
    circle = [cos(theta); sin(theta)];

    ellipse = V * sqrt(D * chi2_val) * circle;
    ellipse(1,:) = ellipse(1,:) + mu(1);
    ellipse(2,:) = ellipse(2,:) + mu(2);

    scatter(x, y, 18, 'filled');
    hold on;
    plot(ellipse(1,:), ellipse(2,:), 'LineWidth', 2);
    plot(mu(1), mu(2), 'rx', 'LineWidth', 2, 'MarkerSize', 10);
    plot(0, 0, 'ko', 'LineWidth', 2, 'MarkerSize', 8);

    grid on;
    axis equal;
    xlabel(xlabel_name);
    ylabel(ylabel_name);
    title(title_name);

    legend('Monte Carlo samples', ...
           '95% covariance ellipse', ...
           'Mean error center', ...
           'True zero error', ...
           'Location', 'best');
end

function angle_deg = min_pairwise_normal_angle_deg(nMat)

    angles = zeros(3,1);
    idx = 1;

    for i = 1:2
        for j = i+1:3
            ni = nMat(:,i) / norm(nMat(:,i));
            nj = nMat(:,j) / norm(nMat(:,j));

            % Use absolute dot product because plane normals have sign ambiguity.
            val = abs(dot(ni, nj));
            val = max(min(val, 1), -1);

            angle = acos(val) * 180 / pi;

            % The angle between plane normals is in [0,90] after abs().
            angles(idx) = angle;
            idx = idx + 1;
        end
    end

    angle_deg = min(angles);
end

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
