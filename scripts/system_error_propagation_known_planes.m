clc;
clear;
close all;

rng(30);

%% ============================================================
%  Systematic Error Propagation for Three-Plane Calibration
%
%  Purpose:
%  After random-error propagation, this script evaluates systematic errors:
%  1) angle zero bias,
%  2) axis non-orthogonality / coordinate skew,
%  3) range scale / temperature-like drift,
%  4) combined systematic + random errors.
%
%  Geometry is fixed:
%  - workspace nominal size: 5 m x 5 m
%  - three boards: 1 m x 0.5 m
%  - 1200 points per board
%  - known plane membership, no RANSAC, no outliers
%
%  Units:
%  - internal length unit: meter
%  - printed position error: millimeter
%  - angle error: degree or microradian
% ============================================================

%% ------------------------------
% Basic settings
%% ------------------------------
num_trials = 300;

workspace_size_m = 5.0;
board_size_1_m = 1.0;
board_size_2_m = 0.5;
num_points_per_board = 1200;

% Measurement distances from scanner to board intersection origin
distance_list_m = [5, 10, 50, 100, 150, 200];

% Random instrument specifications
range_sigma_m = 10e-6;       % 10 um
angle_sigma_rad = 10e-6;     % 10 urad
roughness_sigma_m = 10e-6;   % 10 um

% Systematic-error nominal magnitudes
angle_zero_bias_rad = 10e-6;        % 10 urad fixed angular zero bias
axis_nonorth_rad = 10e-6;           % 10 urad coordinate-axis skew
range_scale_bias_ppm = 10;          % 10 ppm range scale / temperature-like scale drift

% Sensitivity sweep at representative distance
sensitivity_distance_m = 100;
angle_bias_sweep_urad = [1, 5, 10, 20, 50];
axis_nonorth_sweep_urad = [1, 5, 10, 20, 50];
scale_bias_sweep_ppm = [1, 5, 10, 20, 50];

%% ------------------------------
% Fixed three-plane geometry
%% ------------------------------
% Use orthogonal planes as a lower-bound / engineering reference geometry.
n_true = eye(3);

% This offset is lateral/vertical relative to scanner.
origin_lateral_y_m = -0.20;
origin_lateral_z_m = 0.40;

fprintf('================ Fixed Engineering Geometry ================\n');
fprintf('Workspace size: %.2f m x %.2f m\n', workspace_size_m, workspace_size_m);
fprintf('Board size: %.2f m x %.2f m\n', board_size_1_m, board_size_2_m);
fprintf('Points per board: %d\n', num_points_per_board);
fprintf('True plane normals are orthogonal, kappa(N)=%.6f\n', cond(n_true'));
fprintf('Minimum pairwise normal angle = %.6f deg\n', minimum_pairwise_normal_angle_deg(n_true));

fprintf('\n================ Instrument Specifications ================\n');
fprintf('Random range precision sigma_r = %.3f um\n', range_sigma_m * 1e6);
fprintf('Random turntable repeatability sigma_theta/sigma_phi = %.3f urad\n', angle_sigma_rad * 1e6);
fprintf('Plane roughness sigma_rough = %.3f um\n', roughness_sigma_m * 1e6);
fprintf('Systematic angle zero bias = %.3f urad\n', angle_zero_bias_rad * 1e6);
fprintf('Systematic axis non-orthogonality/skew = %.3f urad\n', axis_nonorth_rad * 1e6);
fprintf('Systematic range scale bias = %.3f ppm\n', range_scale_bias_ppm);

%% ============================================================
%  Distance sweep for fixed systematic-error levels
%% ============================================================

case_names = { ...
    'AngleZeroBias_10urad', ...
    'AxisNonorthogonality_10urad', ...
    'RangeScaleBias_10ppm', ...
    'Combined_Systematic', ...
    'Combined_RandomAndSystematic'};

num_cases = numel(case_names);

summary_rows = [];

for c = 1:num_cases

    for d_idx = 1:numel(distance_list_m)

        distance_m = distance_list_m(d_idx);

        fprintf('\n============================================================\n');
        fprintf('Running case: %s, distance = %.1f m\n', case_names{c}, distance_m);
        fprintf('============================================================\n');

        params = default_error_params();

        switch case_names{c}
            case 'AngleZeroBias_10urad'
                params.angle_zero_theta_rad = angle_zero_bias_rad;
                params.angle_zero_phi_rad = angle_zero_bias_rad;

            case 'AxisNonorthogonality_10urad'
                params.axis_nonorth_rad = axis_nonorth_rad;

            case 'RangeScaleBias_10ppm'
                params.range_scale_bias_ppm = range_scale_bias_ppm;

            case 'Combined_Systematic'
                params.angle_zero_theta_rad = angle_zero_bias_rad;
                params.angle_zero_phi_rad = angle_zero_bias_rad;
                params.axis_nonorth_rad = axis_nonorth_rad;
                params.range_scale_bias_ppm = range_scale_bias_ppm;

            case 'Combined_RandomAndSystematic'
                params.use_random_range = true;
                params.use_random_angle = true;
                params.use_roughness = true;
                params.range_sigma_m = range_sigma_m;
                params.angle_sigma_rad = angle_sigma_rad;
                params.roughness_sigma_m = roughness_sigma_m;

                params.angle_zero_theta_rad = angle_zero_bias_rad;
                params.angle_zero_phi_rad = angle_zero_bias_rad;
                params.axis_nonorth_rad = axis_nonorth_rad;
                params.range_scale_bias_ppm = range_scale_bias_ppm;
        end

        result = run_one_condition( ...
            num_trials, distance_m, origin_lateral_y_m, origin_lateral_z_m, ...
            n_true, board_size_1_m, board_size_2_m, num_points_per_board, params);

        row = table( ...
            string(case_names{c}), ...
            distance_m, ...
            params.angle_zero_theta_rad * 1e6, ...
            params.axis_nonorth_rad * 1e6, ...
            params.range_scale_bias_ppm, ...
            params.use_random_range, ...
            params.use_random_angle, ...
            params.use_roughness, ...
            result.mean_point_rms_mm, ...
            result.p95_point_rms_mm, ...
            result.mean_origin_error_mm, ...
            result.p95_origin_error_mm, ...
            result.mean_rotation_error_deg, ...
            result.p95_rotation_error_deg, ...
            result.mean_normal_angle_error_deg, ...
            result.p95_normal_angle_error_deg, ...
            result.mean_axis_orthogonality_error, ...
            result.origin_error_to_point_rms_ratio, ...
            'VariableNames', { ...
            'Case', ...
            'Distance_m', ...
            'Angle_Zero_Bias_urad', ...
            'Axis_Nonorthogonality_urad', ...
            'Range_Scale_Bias_ppm', ...
            'Random_Range_On', ...
            'Random_Angle_On', ...
            'Roughness_On', ...
            'Measured_Point_RMS_mm', ...
            'Measured_Point_P95_mm', ...
            'Mean_Origin_Error_mm', ...
            'P95_Origin_Error_mm', ...
            'Mean_Rotation_Error_deg', ...
            'P95_Rotation_Error_deg', ...
            'Mean_Normal_Angle_Error_deg', ...
            'P95_Normal_Angle_Error_deg', ...
            'Mean_Axis_Orthogonality_Error', ...
            'OriginError_to_PointRMS_Ratio'});

        summary_rows = [summary_rows; row];

        fprintf('Finished case %s, distance %.1f m\n', case_names{c}, distance_m);
        fprintf('Mean measured point RMS = %.6f mm\n', result.mean_point_rms_mm);
        fprintf('Mean origin error = %.6f mm, P95 = %.6f mm\n', ...
            result.mean_origin_error_mm, result.p95_origin_error_mm);
        fprintf('Mean rotation error = %.8f deg, P95 = %.8f deg\n', ...
            result.mean_rotation_error_deg, result.p95_rotation_error_deg);
    end
end

disp(' ');
disp('================ Systematic Error Distance-Sweep Summary ================');
disp(summary_rows);

%% ============================================================
%  Sensitivity sweep at a representative distance
%% ============================================================

sens_rows = [];

fprintf('\n\n================ Sensitivity Sweep at %.1f m ================\n', sensitivity_distance_m);

% 1) Angle zero-bias sweep
for v = angle_bias_sweep_urad
    params = default_error_params();
    params.angle_zero_theta_rad = v * 1e-6;
    params.angle_zero_phi_rad = v * 1e-6;

    result = run_one_condition( ...
        num_trials, sensitivity_distance_m, origin_lateral_y_m, origin_lateral_z_m, ...
        n_true, board_size_1_m, board_size_2_m, num_points_per_board, params);

    sens_rows = [sens_rows; make_sensitivity_row( ...
        'AngleZeroBias', v, 'urad', result)];
end

% 2) Axis non-orthogonality sweep
for v = axis_nonorth_sweep_urad
    params = default_error_params();
    params.axis_nonorth_rad = v * 1e-6;

    result = run_one_condition( ...
        num_trials, sensitivity_distance_m, origin_lateral_y_m, origin_lateral_z_m, ...
        n_true, board_size_1_m, board_size_2_m, num_points_per_board, params);

    sens_rows = [sens_rows; make_sensitivity_row( ...
        'AxisNonorthogonality', v, 'urad', result)];
end

% 3) Range scale-bias sweep
for v = scale_bias_sweep_ppm
    params = default_error_params();
    params.range_scale_bias_ppm = v;

    result = run_one_condition( ...
        num_trials, sensitivity_distance_m, origin_lateral_y_m, origin_lateral_z_m, ...
        n_true, board_size_1_m, board_size_2_m, num_points_per_board, params);

    sens_rows = [sens_rows; make_sensitivity_row( ...
        'RangeScaleBias', v, 'ppm', result)];
end

disp(' ');
disp('================ Systematic Error Sensitivity Summary ================');
disp(sens_rows);

%% ============================================================
%  Simple plots
%% ============================================================

% Combined systematic + random distance sweep
idx_comb = summary_rows.Case == "Combined_RandomAndSystematic";

figure;
plot(summary_rows.Distance_m(idx_comb), summary_rows.Mean_Origin_Error_mm(idx_comb), '-o', 'LineWidth', 2);
grid on;
xlabel('Distance / m');
ylabel('Mean origin error / mm');
title('Combined random + systematic origin error versus distance');

figure;
plot(summary_rows.Distance_m(idx_comb), summary_rows.Mean_Rotation_Error_deg(idx_comb), '-o', 'LineWidth', 2);
grid on;
xlabel('Distance / m');
ylabel('Mean rotation error / deg');
title('Combined random + systematic rotation error versus distance');

% Sensitivity plot at representative distance
% Use row index instead of categorical x-axis because Error_Source has repeated labels.
figure;
bar(1:height(sens_rows), sens_rows.Mean_Origin_Error_mm);
grid on;
ylabel('Mean origin error / mm');
title(sprintf('Systematic error sensitivity at %.1f m', sensitivity_distance_m));
xticks(1:height(sens_rows));
xticklabels(strcat(sens_rows.Error_Source, "_", string(sens_rows.Error_Value), sens_rows.Error_Unit));
xtickangle(45);

% Grouped sensitivity plot: one curve for each systematic error source
figure;
hold on;
grid on;
sources = unique(sens_rows.Error_Source, 'stable');
for si = 1:numel(sources)
    idx = sens_rows.Error_Source == sources(si);
    plot(sens_rows.Error_Value(idx), sens_rows.Mean_Origin_Error_mm(idx), '-o', 'LineWidth', 2);
end
xlabel('Error magnitude / urad or ppm');
ylabel('Mean origin error / mm');
title(sprintf('Systematic error sensitivity curves at %.1f m', sensitivity_distance_m));
legend(sources, 'Location', 'northwest');

save('system_error_propagation_known_planes_results.mat', ...
    'summary_rows', 'sens_rows');

fprintf('\nSaved results to system_error_propagation_known_planes_results.mat\n');

%% ============================================================
%  Local functions
%% ============================================================

function params = default_error_params()
    params.use_random_range = false;
    params.use_random_angle = false;
    params.use_roughness = false;

    params.range_sigma_m = 0;
    params.angle_sigma_rad = 0;
    params.roughness_sigma_m = 0;

    params.angle_zero_theta_rad = 0;
    params.angle_zero_phi_rad = 0;
    params.axis_nonorth_rad = 0;
    params.range_scale_bias_ppm = 0;
end

function row = make_sensitivity_row(source_name, value, unit_name, result)
    row = table( ...
        string(source_name), ...
        value, ...
        string(unit_name), ...
        result.mean_point_rms_mm, ...
        result.mean_origin_error_mm, ...
        result.p95_origin_error_mm, ...
        result.mean_rotation_error_deg, ...
        result.p95_rotation_error_deg, ...
        result.mean_normal_angle_error_deg, ...
        result.mean_axis_orthogonality_error, ...
        'VariableNames', { ...
        'Error_Source', ...
        'Error_Value', ...
        'Error_Unit', ...
        'Measured_Point_RMS_mm', ...
        'Mean_Origin_Error_mm', ...
        'P95_Origin_Error_mm', ...
        'Mean_Rotation_Error_deg', ...
        'P95_Rotation_Error_deg', ...
        'Mean_Normal_Angle_Error_deg', ...
        'Mean_Axis_Orthogonality_Error'});
end

function result = run_one_condition( ...
    num_trials, distance_m, lateral_y_m, lateral_z_m, ...
    n_true, board_size_1_m, board_size_2_m, num_points_per_board, params)

    d_true = zeros(3,1);

    % Plane intersection origin at approximately given distance from scanner.
    O_true = [distance_m; lateral_y_m; lateral_z_m];

    for k = 1:3
        d_true(k) = -n_true(:,k)' * O_true;
    end

    [O_check, R_true] = frame_from_three_planes(n_true, d_true);
    if norm(O_check - O_true) > 1e-9
        error('Internal geometry check failed.');
    end

    origin_error_mm = nan(num_trials,1);
    rotation_error_deg_arr = nan(num_trials,1);
    normal_angle_error_deg = nan(num_trials,1);
    axis_orthogonality_error = nan(num_trials,1);
    point_rms_mm = nan(num_trials,1);
    point_p95_mm = nan(num_trials,1);

    for trial = 1:num_trials

        P_all_true = [];
        P_all_meas = [];
        fitted_n = zeros(3,3);
        fitted_d = zeros(3,1);

        for k = 1:3
            P_true = sample_points_on_axis_aligned_plane( ...
                k, O_true, board_size_1_m, board_size_2_m, num_points_per_board);

            P_phys = P_true;

            % Physical plane roughness: displacement along true plane normal.
            if params.use_roughness
                rough = params.roughness_sigma_m * randn(num_points_per_board,1);
                P_phys = P_phys + rough * n_true(:,k)';
            end

            P_meas = apply_measurement_errors(P_phys, params);

            [n_k, d_k] = fit_plane_svd(P_meas);

            % Align normal sign with truth.
            if dot(n_k, n_true(:,k)) < 0
                n_k = -n_k;
                d_k = -d_k;
            end

            fitted_n(:,k) = n_k;
            fitted_d(k) = d_k;

            P_all_true = [P_all_true; P_true];
            P_all_meas = [P_all_meas; P_meas];
        end

        [O_est, R_est] = frame_from_three_planes(fitted_n, fitted_d);

        origin_error_mm(trial) = norm(O_est - O_true) * 1000;
        rotation_error_deg_arr(trial) = rotation_error_deg(R_est, R_true);

        ne = zeros(3,1);
        for k = 1:3
            ne(k) = acosd_clamped(abs(dot(fitted_n(:,k), n_true(:,k))));
        end
        normal_angle_error_deg(trial) = mean(ne);

        axis_orthogonality_error(trial) = raw_axis_orthogonality_error(fitted_n);

        point_err = sqrt(sum((P_all_meas - P_all_true).^2, 2)) * 1000;
        point_rms_mm(trial) = sqrt(mean(point_err.^2));
        point_p95_mm(trial) = percentile_simple(point_err, 95);
    end

    result.mean_origin_error_mm = mean(origin_error_mm);
    result.p95_origin_error_mm = percentile_simple(origin_error_mm, 95);
    result.std_origin_error_mm = std(origin_error_mm);

    result.mean_rotation_error_deg = mean(rotation_error_deg_arr);
    result.p95_rotation_error_deg = percentile_simple(rotation_error_deg_arr, 95);

    result.mean_normal_angle_error_deg = mean(normal_angle_error_deg);
    result.p95_normal_angle_error_deg = percentile_simple(normal_angle_error_deg, 95);

    result.mean_axis_orthogonality_error = mean(axis_orthogonality_error);

    result.mean_point_rms_mm = mean(point_rms_mm);
    result.p95_point_rms_mm = percentile_simple(point_p95_mm, 95);

    if result.mean_point_rms_mm > 0
        result.origin_error_to_point_rms_ratio = ...
            result.mean_origin_error_mm / result.mean_point_rms_mm;
    else
        result.origin_error_to_point_rms_ratio = NaN;
    end
end

function P = sample_points_on_axis_aligned_plane(plane_id, O, size1, size2, N)
    % Axis-aligned finite boards intersect at O.
    % plane 1: x = O_x, board spans y-z
    % plane 2: y = O_y, board spans x-z
    % plane 3: z = O_z, board spans x-y

    u = (rand(N,1) - 0.5) * size1;
    v = (rand(N,1) - 0.5) * size2;

    switch plane_id
        case 1
            P = [O(1) * ones(N,1), O(2) + u, O(3) + v];
        case 2
            P = [O(1) + u, O(2) * ones(N,1), O(3) + v];
        case 3
            P = [O(1) + u, O(2) + v, O(3) * ones(N,1)];
        otherwise
            error('Invalid plane_id.');
    end
end

function P_meas = apply_measurement_errors(P, params)
    x = P(:,1);
    y = P(:,2);
    z = P(:,3);

    r = sqrt(x.^2 + y.^2 + z.^2);
    theta = atan2(y, x);
    phi = asin(z ./ r);

    % Systematic range scale.
    r = r .* (1 + params.range_scale_bias_ppm * 1e-6);

    % Random range.
    if params.use_random_range
        r = r + params.range_sigma_m * randn(size(r));
    end

    % Systematic angle zero bias.
    theta = theta + params.angle_zero_theta_rad;
    phi = phi + params.angle_zero_phi_rad;

    % Random angle repeatability.
    if params.use_random_angle
        theta = theta + params.angle_sigma_rad * randn(size(theta));
        phi = phi + params.angle_sigma_rad * randn(size(phi));
    end

    % Back to Cartesian.
    x2 = r .* cos(phi) .* cos(theta);
    y2 = r .* cos(phi) .* sin(theta);
    z2 = r .* sin(phi);

    P_meas = [x2, y2, z2];

    % Axis non-orthogonality / coordinate skew.
    eps = params.axis_nonorth_rad;
    if eps ~= 0
        % A simple first-order skew model. This approximates coupling between
        % nominal coordinate axes caused by a small non-orthogonal axis setup.
        S = [1, eps, 0;
             0, 1, eps;
             0, 0, 1];
        P_meas = (S * P_meas')';
    end
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

    e2_temp = n2 - dot(e1, n2) * e1;

    if norm(e2_temp) < 1e-12
        tmp = [1;0;0];
        if abs(dot(tmp,e1)) > 0.9
            tmp = [0;1;0];
        end
        e2_temp = tmp - dot(tmp,e1) * e1;
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
    value = (trace(R_err)-1)/2;
    value = max(min(value,1),-1);
    err = acos(value) * 180 / pi;
end

function e = raw_axis_orthogonality_error(nMat)
    n1 = nMat(:,1) / norm(nMat(:,1));
    n2 = nMat(:,2) / norm(nMat(:,2));
    n3 = nMat(:,3) / norm(nMat(:,3));

    e = mean(abs([dot(n1,n2), dot(n1,n3), dot(n2,n3)]));
end

function a = acosd_clamped(x)
    x = max(min(x,1),-1);
    a = acos(x) * 180 / pi;
end

function min_angle = minimum_pairwise_normal_angle_deg(nMat)
    angles = [];
    for i = 1:3
        for j = i+1:3
            ni = nMat(:,i) / norm(nMat(:,i));
            nj = nMat(:,j) / norm(nMat(:,j));
            angles(end+1) = acosd_clamped(abs(dot(ni,nj))); %#ok<AGROW>
        end
    end
    min_angle = min(angles);
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
    rank = 1 + (percent/100) * (n-1);
    low = floor(rank);
    high = ceil(rank);
    if low == high
        p = x(low);
    else
        w = rank - low;
        p = (1-w) * x(low) + w * x(high);
    end
end
