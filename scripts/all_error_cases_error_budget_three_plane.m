clc;
clear;
close all;

rng(40);

%% ============================================================
%  All Error Cases + Error Budget Table
%  for Three-Plane Calibration
%
%  Purpose:
%  1. Use fixed engineering geometry:
%       - workspace: 5 m x 5 m
%       - three boards: 1 m x 0.5 m
%       - >1000 valid points per board
%       - known board membership, no RANSAC, no outliers
%
%  2. Compare all major error cases:
%       - plane roughness
%       - range random error
%       - turntable random angle error
%       - turntable angle zero bias
%       - two-axis non-orthogonality
%       - laser boresight pointing bias
%       - laser/turntable center offset, first-order lever-arm model
%       - thermal scale drift
%       - air refractive-index scale bias
%       - combined random errors
%       - combined systematic errors
%       - combined all errors
%
%  3. Automatically generate an error budget table at 100 m.
%
%  Notes:
%  - Internal length unit: meter.
%  - Output origin error: millimeter.
%  - Output rotation error: degree.
%  - This script is a first-order sensitivity model. More detailed optical-
%    mechanical models can replace each case later.
% ============================================================

%% ============================================================
%  Basic settings
%% ============================================================

num_trials = 300;

workspace_size_m = 5.0;
board_size_1_m = 1.0;
board_size_2_m = 0.5;
num_points_per_board = 1200;

distance_list_m = [5, 10, 50, 100, 150, 200];
budget_distance_m = 100;

% Fixed orthogonal reference geometry.
% n1 = x-axis normal, n2 = y-axis normal, n3 = z-axis normal
n_true = eye(3);

% Lateral/vertical location of the plane-intersection origin relative to scanner.
origin_lateral_y_m = -0.20;
origin_lateral_z_m = 0.40;

%% ============================================================
%  Representative error values
%% ============================================================

% Random errors
range_sigma_m = 10e-6;             % 10 um
angle_random_sigma_rad = 10e-6;    % 10 urad
roughness_sigma_m = 10e-6;         % 10 um

% Systematic errors
angle_zero_bias_rad = 10e-6;       % 10 urad
axis_nonorth_rad = 10e-6;          % 10 urad
beam_boresight_bias_rad = 10e-6;   % 10 urad
beam_axis_offset_m = 1e-3;         % 1 mm first-order center/lever-arm offset
thermal_scale_bias_ppm = 10;       % 10 ppm
refractive_index_bias_ppm = 10;    % 10 ppm

fprintf('================ Fixed Engineering Geometry ================\n');
fprintf('Workspace size: %.2f m x %.2f m\n', workspace_size_m, workspace_size_m);
fprintf('Board size: %.2f m x %.2f m\n', board_size_1_m, board_size_2_m);
fprintf('Points per board: %d\n', num_points_per_board);
fprintf('True plane normals are orthogonal, kappa(N)=%.6f\n', cond(n_true'));
fprintf('Minimum pairwise normal angle = %.6f deg\n', minimum_pairwise_normal_angle_deg(n_true));

fprintf('\n================ Representative Error Values ================\n');
fprintf('Plane roughness sigma: %.3f um\n', roughness_sigma_m * 1e6);
fprintf('Range random sigma: %.3f um\n', range_sigma_m * 1e6);
fprintf('Turntable random angle sigma: %.3f urad\n', angle_random_sigma_rad * 1e6);
fprintf('Turntable angle zero bias: %.3f urad\n', angle_zero_bias_rad * 1e6);
fprintf('Axis non-orthogonality: %.3f urad\n', axis_nonorth_rad * 1e6);
fprintf('Laser boresight bias: %.3f urad\n', beam_boresight_bias_rad * 1e6);
fprintf('Laser/turntable center offset: %.3f mm\n', beam_axis_offset_m * 1000);
fprintf('Thermal scale drift: %.3f ppm\n', thermal_scale_bias_ppm);
fprintf('Refractive-index scale bias: %.3f ppm\n', refractive_index_bias_ppm);

%% ============================================================
%  Define all error cases
%% ============================================================

cases = {};

% ------------------------------
% Random-only cases
% ------------------------------
cases{end+1} = make_case('PlaneRoughnessOnly_10um', ...
    'Random', 'Plane roughness / flatness noise', '10 um', ...
    true, false, false, ...
    roughness_sigma_m, 0, 0, ...
    0, 0, 0, 0, 0, 0, ...
    'Can be reduced by multi-point plane fitting');

cases{end+1} = make_case('RangeRandomOnly_10um', ...
    'Random', 'Range measurement random noise', '10 um', ...
    false, true, false, ...
    0, range_sigma_m, 0, ...
    0, 0, 0, 0, 0, 0, ...
    'Can be reduced by repeated measurements and plane fitting');

cases{end+1} = make_case('AngleRandomOnly_10urad', ...
    'Random', 'Turntable repeatability / random angular noise', '10 urad', ...
    false, false, true, ...
    0, 0, angle_random_sigma_rad, ...
    0, 0, 0, 0, 0, 0, ...
    'Can be reduced by dense sampling, repeated scans, and averaging');

cases{end+1} = make_case('Combined_Random', ...
    'Random', 'Roughness + range random + angle random', '10um, 10um, 10urad', ...
    true, true, true, ...
    roughness_sigma_m, range_sigma_m, angle_random_sigma_rad, ...
    0, 0, 0, 0, 0, 0, ...
    'Reduced by plane fitting; dominated by angular random error at long range');

% ------------------------------
% Systematic-only cases
% ------------------------------
cases{end+1} = make_case('AngleZeroBias_10urad', ...
    'Systematic', 'Turntable angular zero bias', '10 urad', ...
    false, false, false, ...
    0, 0, 0, ...
    angle_zero_bias_rad, 0, 0, 0, 0, 0, ...
    'Requires angular zero calibration');

cases{end+1} = make_case('AxisNonorthogonality_10urad', ...
    'Systematic', 'Two-axis non-orthogonality / axis skew', '10 urad', ...
    false, false, false, ...
    0, 0, 0, ...
    0, axis_nonorth_rad, 0, 0, 0, 0, ...
    'Requires mechanical axis calibration / axis model compensation');

cases{end+1} = make_case('BeamBoresightBias_10urad', ...
    'Systematic', 'Laser optical-axis pointing bias', '10 urad', ...
    false, false, false, ...
    0, 0, 0, ...
    0, 0, beam_boresight_bias_rad, 0, 0, 0, ...
    'Requires boresight calibration between laser beam and scanner axes');

cases{end+1} = make_case('BeamAxisOffset_1mm', ...
    'Systematic', 'Laser emission center offset from turntable center', '1 mm', ...
    false, false, false, ...
    0, 0, 0, ...
    0, 0, 0, beam_axis_offset_m, 0, 0, ...
    'Requires lever-arm / center-offset calibration');

cases{end+1} = make_case('ThermalScaleBias_10ppm', ...
    'Systematic', 'Temperature-induced range or structure scale drift', '10 ppm', ...
    false, false, false, ...
    0, 0, 0, ...
    0, 0, 0, 0, thermal_scale_bias_ppm, 0, ...
    'Requires temperature monitoring and scale compensation');

cases{end+1} = make_case('RefractiveIndexBias_10ppm', ...
    'Systematic', 'Air refractive-index range scale error', '10 ppm', ...
    false, false, false, ...
    0, 0, 0, ...
    0, 0, 0, 0, 0, refractive_index_bias_ppm, ...
    'Requires refractive-index / environmental compensation');

cases{end+1} = make_case('Combined_Systematic', ...
    'Systematic', 'All systematic errors combined', ...
    '10urad,10urad,10urad,1mm,10ppm,10ppm', ...
    false, false, false, ...
    0, 0, 0, ...
    angle_zero_bias_rad, axis_nonorth_rad, beam_boresight_bias_rad, ...
    beam_axis_offset_m, thermal_scale_bias_ppm, refractive_index_bias_ppm, ...
    'Requires full system calibration and compensation');

% ------------------------------
% All-error case
% ------------------------------
cases{end+1} = make_case('Combined_AllErrors', ...
    'Mixed', 'All random and systematic errors combined', ...
    'all representative values', ...
    true, true, true, ...
    roughness_sigma_m, range_sigma_m, angle_random_sigma_rad, ...
    angle_zero_bias_rad, axis_nonorth_rad, beam_boresight_bias_rad, ...
    beam_axis_offset_m, thermal_scale_bias_ppm, refractive_index_bias_ppm, ...
    'Full error budget case');

%% ============================================================
%  Run distance sweep
%% ============================================================

all_rows = table();

for ci = 1:numel(cases)

    current_case = cases{ci};

    for di = 1:numel(distance_list_m)
        distance_m = distance_list_m(di);

        fprintf('\n============================================================\n');
        fprintf('Running case: %s, distance = %.1f m\n', current_case.name, distance_m);
        fprintf('============================================================\n');

        result = run_one_condition( ...
            num_trials, distance_m, origin_lateral_y_m, origin_lateral_z_m, ...
            n_true, board_size_1_m, board_size_2_m, num_points_per_board, current_case);

        row = table( ...
            string(current_case.name), ...
            string(current_case.error_type), ...
            string(current_case.description), ...
            string(current_case.representative_value), ...
            distance_m, ...
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
            string(current_case.mitigation), ...
            'VariableNames', { ...
            'Case', ...
            'Error_Type', ...
            'Physical_Meaning', ...
            'Representative_Value', ...
            'Distance_m', ...
            'Measured_Point_RMS_mm', ...
            'Measured_Point_P95_mm', ...
            'Mean_Origin_Error_mm', ...
            'P95_Origin_Error_mm', ...
            'Mean_Rotation_Error_deg', ...
            'P95_Rotation_Error_deg', ...
            'Mean_Normal_Angle_Error_deg', ...
            'P95_Normal_Angle_Error_deg', ...
            'Mean_Axis_Orthogonality_Error', ...
            'OriginError_to_PointRMS_Ratio', ...
            'Mitigation'});

        all_rows = [all_rows; row];

        fprintf('Mean point RMS = %.6f mm\n', result.mean_point_rms_mm);
        fprintf('Mean origin error = %.6f mm, P95 = %.6f mm\n', ...
            result.mean_origin_error_mm, result.p95_origin_error_mm);
        fprintf('Mean rotation error = %.8f deg, P95 = %.8f deg\n', ...
            result.mean_rotation_error_deg, result.p95_rotation_error_deg);
    end
end

disp(' ');
disp('================ All Error Cases Distance-Sweep Summary ================');
disp(all_rows);

%% ============================================================
%  Generate error budget table at selected distance
%% ============================================================

idx_budget = abs(all_rows.Distance_m - budget_distance_m) < 1e-9;
error_budget = all_rows(idx_budget, :);

% Sort by mean origin error descending.
[~, sort_idx] = sort(error_budget.Mean_Origin_Error_mm, 'descend');
error_budget = error_budget(sort_idx, :);

% Add rank.
Rank = (1:height(error_budget))';
error_budget = addvars(error_budget, Rank, 'Before', 'Case');

disp(' ');
fprintf('================ Error Budget Table at %.1f m ================\n', budget_distance_m);
disp(error_budget);

% Compact budget table for reports.
compact_budget = error_budget(:, { ...
    'Rank', ...
    'Case', ...
    'Error_Type', ...
    'Physical_Meaning', ...
    'Representative_Value', ...
    'Mean_Origin_Error_mm', ...
    'P95_Origin_Error_mm', ...
    'Mean_Rotation_Error_deg', ...
    'P95_Rotation_Error_deg', ...
    'Mitigation'});

disp(' ');
fprintf('================ Compact Error Budget for Report at %.1f m ================\n', budget_distance_m);
disp(compact_budget);

%% ============================================================
%  Plots
%% ============================================================

% 1) Budget bar chart at representative distance.
figure;
bar(1:height(compact_budget), compact_budget.Mean_Origin_Error_mm);
grid on;
ylabel('Mean origin error / mm');
title(sprintf('Error budget ranking at %.1f m', budget_distance_m));
xticks(1:height(compact_budget));
xticklabels(compact_budget.Case);
xtickangle(45);

% 2) Distance sweep for selected important cases.
important_names = ["Combined_AllErrors", "Combined_Systematic", "Combined_Random", ...
                   "AngleZeroBias_10urad", "BeamBoresightBias_10urad", ...
                   "ThermalScaleBias_10ppm", "RefractiveIndexBias_10ppm", ...
                   "BeamAxisOffset_1mm"];

figure;
hold on;
grid on;
for k = 1:numel(important_names)
    idx = all_rows.Case == important_names(k);
    if any(idx)
        plot(all_rows.Distance_m(idx), all_rows.Mean_Origin_Error_mm(idx), '-o', 'LineWidth', 2);
    end
end
xlabel('Distance / m');
ylabel('Mean origin error / mm');
title('Mean origin error versus distance for major error sources');
legend(important_names, 'Location', 'northwest');

% 3) Rotation error sweep.
figure;
hold on;
grid on;
for k = 1:numel(important_names)
    idx = all_rows.Case == important_names(k);
    if any(idx)
        plot(all_rows.Distance_m(idx), all_rows.Mean_Rotation_Error_deg(idx), '-o', 'LineWidth', 2);
    end
end
xlabel('Distance / m');
ylabel('Mean rotation error / deg');
title('Mean rotation error versus distance for major error sources');
legend(important_names, 'Location', 'northwest');

save('all_error_cases_error_budget_results.mat', ...
    'all_rows', 'error_budget', 'compact_budget');

fprintf('\nSaved results to all_error_cases_error_budget_results.mat\n');

%% ============================================================
%  Local functions
%% ============================================================

function c = make_case(name, error_type, description, rep_value, ...
    use_roughness, use_range_random, use_angle_random, ...
    roughness_sigma_m, range_sigma_m, angle_random_sigma_rad, ...
    angle_zero_bias_rad, axis_nonorth_rad, beam_boresight_bias_rad, ...
    beam_axis_offset_m, thermal_scale_bias_ppm, refractive_index_bias_ppm, mitigation)

    c.name = name;
    c.error_type = error_type;
    c.description = description;
    c.representative_value = rep_value;

    c.use_roughness = use_roughness;
    c.use_range_random = use_range_random;
    c.use_angle_random = use_angle_random;

    c.roughness_sigma_m = roughness_sigma_m;
    c.range_sigma_m = range_sigma_m;
    c.angle_random_sigma_rad = angle_random_sigma_rad;

    c.angle_zero_bias_rad = angle_zero_bias_rad;
    c.axis_nonorth_rad = axis_nonorth_rad;
    c.beam_boresight_bias_rad = beam_boresight_bias_rad;
    c.beam_axis_offset_m = beam_axis_offset_m;
    c.thermal_scale_bias_ppm = thermal_scale_bias_ppm;
    c.refractive_index_bias_ppm = refractive_index_bias_ppm;

    c.mitigation = mitigation;
end

function result = run_one_condition( ...
    num_trials, distance_m, lateral_y_m, lateral_z_m, ...
    n_true, board_size_1_m, board_size_2_m, num_points_per_board, c)

    O_true = [distance_m; lateral_y_m; lateral_z_m];

    d_true = zeros(3,1);
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

        fitted_n = zeros(3,3);
        fitted_d = zeros(3,1);

        P_all_true = [];
        P_all_meas = [];

        for plane_id = 1:3
            P_true = sample_points_on_axis_aligned_plane( ...
                plane_id, O_true, board_size_1_m, board_size_2_m, num_points_per_board);

            P_phys = P_true;

            % Plane roughness / flatness error: along true plane normal.
            if c.use_roughness
                rough = c.roughness_sigma_m * randn(num_points_per_board, 1);
                P_phys = P_phys + rough * n_true(:,plane_id)';
            end

            P_meas = apply_all_errors(P_phys, c);

            [n_k, d_k] = fit_plane_svd(P_meas);

            % Align estimated normal with true normal for evaluation.
            if dot(n_k, n_true(:,plane_id)) < 0
                n_k = -n_k;
                d_k = -d_k;
            end

            fitted_n(:,plane_id) = n_k;
            fitted_d(plane_id) = d_k;

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

        point_err_mm = sqrt(sum((P_all_meas - P_all_true).^2, 2)) * 1000;
        point_rms_mm(trial) = sqrt(mean(point_err_mm.^2));
        point_p95_mm(trial) = percentile_simple(point_err_mm, 95);
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

function P_meas = apply_all_errors(P, c)

    x = P(:,1);
    y = P(:,2);
    z = P(:,3);

    r = sqrt(x.^2 + y.^2 + z.^2);
    theta = atan2(y, x);
    phi = asin(z ./ r);

    % Scale-like system errors.
    % Thermal scale and refractive-index error are separated physically,
    % but they are both first-order range scale terms in this simplified model.
    total_scale_ppm = c.thermal_scale_bias_ppm + c.refractive_index_bias_ppm;
    r = r .* (1 + total_scale_ppm * 1e-6);

    % Random range error.
    if c.use_range_random
        r = r + c.range_sigma_m * randn(size(r));
    end

    % Turntable angle zero bias.
    theta = theta + c.angle_zero_bias_rad;
    phi = phi + c.angle_zero_bias_rad;

    % Laser boresight pointing bias.
    % Mathematically similar to an angular bias, but kept separate as a
    % physical source.
    theta = theta + c.beam_boresight_bias_rad;
    phi = phi + c.beam_boresight_bias_rad;

    % Random angle repeatability.
    if c.use_angle_random
        theta = theta + c.angle_random_sigma_rad * randn(size(theta));
        phi = phi + c.angle_random_sigma_rad * randn(size(phi));
    end

    % Convert back to Cartesian.
    x2 = r .* cos(phi) .* cos(theta);
    y2 = r .* cos(phi) .* sin(theta);
    z2 = r .* sin(phi);

    P_meas = [x2, y2, z2];

    % Beam-axis offset / laser-turntable center offset.
    % First-order lever-arm model: if a fixed beam origin offset is not
    % compensated, reconstructed points are shifted by this vector.
    % Here we place the representative offset in the transverse y-direction.
    if c.beam_axis_offset_m ~= 0
        lever = [0, c.beam_axis_offset_m, 0];
        P_meas = P_meas - lever;
    end

    % Axis non-orthogonality / coordinate skew.
    eps = c.axis_nonorth_rad;
    if eps ~= 0
        S = [1, eps, 0;
             0, 1, eps;
             0, 0, 1];
        P_meas = (S * P_meas')';
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

    rank = 1 + (percent/100) * (n - 1);
    low = floor(rank);
    high = ceil(rank);

    if low == high
        p = x(low);
    else
        w = rank - low;
        p = (1-w) * x(low) + w * x(high);
    end
end
