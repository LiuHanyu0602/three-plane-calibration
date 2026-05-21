clc;
clear;
close all;

rng(20);

%% ============================================================
%  Instrument-Error Propagation Experiment for Three-Plane Calibration
%
%  Purpose:
%  This script is the next step after the ideal-plane fitting lower-bound
%  experiment. It keeps the engineering geometry fixed, assumes that the
%  three boards are already correctly segmented, and studies how realistic
%  measurement errors propagate to:
%       1) fitted plane normal error,
%       2) fitted plane offset error,
%       3) three-plane origin error,
%       4) rotation matrix error.
%
%  Geometry:
%       workspace: 5 m x 5 m
%       board size: 1 m x 0.5 m
%       points per board: 1200
%
%  Instrument error model:
%       range precision: 10 um
%       two-axis turntable repeatability: 10 urad per angular axis
%       board roughness / plane roughness: 10 um
%
%  Important:
%       This script does NOT use RANSAC and does NOT add outliers.
%       It estimates the best-case propagation under known board labels.
% =============================================================

%% ============================================================
%  Fixed engineering geometry
% =============================================================

cfg.num_trials = 300;
cfg.workspace_size_m = 5.0;
cfg.board_size_1_m = 1.0;
cfg.board_size_2_m = 0.5;
cfg.num_points_per_board = 1200;

% Realistic instrument specifications
cfg.range_sigma_m = 10e-6;        % 10 um range precision
cfg.angle_sigma_rad = 10e-6;      % 10 urad turntable repeatability
cfg.roughness_sigma_m = 10e-6;    % 10 um plane roughness / flatness proxy

% Measurement distances to be tested.
% The whole three-board structure is translated to these ranges from scanner.
distance_list_m = [5, 10, 50, 100, 150, 200];

% Good three-plane geometry: an orthonormal triad rotated in scanner frame.
yaw   = deg2rad(25);
pitch = deg2rad(-12);
roll  = deg2rad(7);
R0 = eulzyx(yaw, pitch, roll);
n_true_template = R0;  % columns are the three true plane normals

% Plane patch centers relative to the three-plane intersection.
% Each column gives two tangent coordinates for one board center.
% Values are kept within the 5 m x 5 m engineering space.
tangent_shift_list = [ 0.20, -0.25,  0.15;
                      -0.10,  0.18,  0.12];

% Error cases for sensitivity analysis.
error_cases = { ...
    make_error_case('RoughnessOnly_10um', true,  false, false), ...
    make_error_case('RangeOnly_10um',     false, true,  false), ...
    make_error_case('AngleOnly_10urad',   false, false, true), ...
    make_error_case('Combined_AllRandom', true,  true,  true)  ...
    };

%% ============================================================
%  Print geometry and instrument settings
% =============================================================

fprintf('\n================ Fixed Engineering Geometry ================\n');
fprintf('Workspace size: %.2f m x %.2f m\n', cfg.workspace_size_m, cfg.workspace_size_m);
fprintf('Board size: %.2f m x %.2f m\n', cfg.board_size_1_m, cfg.board_size_2_m);
fprintf('Points per board: %d\n', cfg.num_points_per_board);
fprintf('Condition number kappa(N): %.6f\n', cond(n_true_template'));
fprintf('Minimum pairwise normal angle: %.6f deg\n', min_pairwise_normal_angle_deg(n_true_template));

fprintf('\n================ Instrument Specifications ================\n');
fprintf('Range precision sigma_r: %.3f um\n', cfg.range_sigma_m * 1e6);
fprintf('Turntable repeatability sigma_theta/sigma_phi: %.3f urad\n', cfg.angle_sigma_rad * 1e6);
fprintf('Plane roughness sigma_rough: %.3f um\n', cfg.roughness_sigma_m * 1e6);

fprintf('\nEquivalent single-axis transverse error: sigma_perp = R * sigma_angle\n');
for i = 1:length(distance_list_m)
    Rm = distance_list_m(i);
    fprintf('  R = %6.1f m: single-axis transverse = %.6f mm, double-axis equivalent = %.6f mm\n', ...
        Rm, Rm * cfg.angle_sigma_rad * 1000, ...
        sqrt(cfg.range_sigma_m^2 + 2 * (Rm * cfg.angle_sigma_rad)^2) * 1000);
end

%% ============================================================
%  Main experiment loop
% =============================================================

summary_rows = [];
raw_results = struct();
row_id = 0;

for c = 1:length(error_cases)

    err_case = error_cases{c};
    case_name = err_case.name;

    for d_idx = 1:length(distance_list_m)

        scene_distance_m = distance_list_m(d_idx);

        fprintf('\n============================================================\n');
        fprintf('Running case: %s, distance = %.1f m\n', case_name, scene_distance_m);
        fprintf('============================================================\n');

        origin_error_mm = nan(cfg.num_trials, 1);
        rotation_error_deg_list = nan(cfg.num_trials, 1);
        normal_angle_error_deg = nan(cfg.num_trials, 1);
        abs_d_error_um = nan(cfg.num_trials, 1);
        raw_normal_orthogonality = nan(cfg.num_trials, 1);
        measured_point_rms_mm = nan(cfg.num_trials, 1);
        measured_point_p95_mm = nan(cfg.num_trials, 1);

        for trial = 1:cfg.num_trials

            % Three-plane intersection point at the desired measurement range.
            % A small offset keeps the structure away from exact scanner axis.
            O_true = [scene_distance_m; 0.20; 0.80];
            n_true = n_true_template;

            d_true = zeros(3,1);
            for k = 1:3
                d_true(k) = -n_true(:,k)' * O_true;
            end

            [O_check, R_true_frame] = frame_from_three_planes(n_true, d_true);
            if norm(O_check - O_true) > 1e-9
                error('True plane reconstruction failed.');
            end

            n_est = zeros(3,3);
            d_est = zeros(3,1);

            all_true_points = [];
            all_measured_points = [];

            for k = 1:3

                P_ideal = sample_points_on_plane( ...
                    n_true(:,k), O_true, tangent_shift_list(:,k), ...
                    cfg.num_points_per_board, cfg.board_size_1_m, cfg.board_size_2_m);

                P_true = P_ideal;

                % Board roughness is modeled as a normal-direction displacement.
                % This represents ideal-board flatness / surface roughness.
                if err_case.use_roughness
                    rough = cfg.roughness_sigma_m * randn(size(P_true,1), 1);
                    P_true = P_true + rough * n_true(:,k)';
                end

                % Instrument measurement error is applied in spherical coordinates.
                P_meas = apply_spherical_instrument_noise( ...
                    P_true, ...
                    cfg.range_sigma_m, cfg.angle_sigma_rad, cfg.angle_sigma_rad, ...
                    err_case.use_range, err_case.use_angle);

                [n_k, d_k] = fit_plane_svd(P_meas);

                % Align normal sign with true normal for fair evaluation.
                if dot(n_k, n_true(:,k)) < 0
                    n_k = -n_k;
                    d_k = -d_k;
                end

                n_est(:,k) = n_k;
                d_est(k) = d_k;

                all_true_points = [all_true_points; P_ideal];
                all_measured_points = [all_measured_points; P_meas];
            end

            [O_est, R_est_frame] = frame_from_three_planes(n_est, d_est);

            origin_error_mm(trial) = norm(O_est - O_true) * 1000;
            rotation_error_deg_list(trial) = rotation_error_deg(R_est_frame, R_true_frame);

            normal_err_each = zeros(3,1);
            d_err_each = zeros(3,1);
            for k = 1:3
                normal_err_each(k) = acosd_safe(abs(dot(n_est(:,k), n_true(:,k))));
                d_err_each(k) = abs(d_est(k) - d_true(k));
            end

            normal_angle_error_deg(trial) = mean(normal_err_each);
            abs_d_error_um(trial) = mean(d_err_each) * 1e6;

            raw_normal_orthogonality(trial) = mean([ ...
                abs(dot(n_est(:,1), n_est(:,2))), ...
                abs(dot(n_est(:,1), n_est(:,3))), ...
                abs(dot(n_est(:,2), n_est(:,3))) ]);

            point_err_mm = vecnorm(all_measured_points - all_true_points, 2, 2) * 1000;
            measured_point_rms_mm(trial) = sqrt(mean(point_err_mm.^2));
            measured_point_p95_mm(trial) = percentile_simple(point_err_mm, 95);
        end

        row_id = row_id + 1;

        row = struct();
        row.Case = string(case_name);
        row.Distance_m = scene_distance_m;
        row.Range_sigma_um = cfg.range_sigma_m * 1e6;
        row.Angle_sigma_urad = cfg.angle_sigma_rad * 1e6;
        row.Roughness_sigma_um = cfg.roughness_sigma_m * 1e6;
        row.Equivalent_DoubleAxis_PointSigma_mm = ...
            sqrt(cfg.range_sigma_m^2 + 2 * (scene_distance_m * cfg.angle_sigma_rad)^2) * 1000;
        row.Measured_Point_RMS_mm = mean(measured_point_rms_mm);
        row.Measured_Point_P95_mm = mean(measured_point_p95_mm);
        row.Mean_Origin_Error_mm = mean(origin_error_mm);
        row.P95_Origin_Error_mm = percentile_simple(origin_error_mm, 95);
        row.Mean_Rotation_Error_deg = mean(rotation_error_deg_list);
        row.P95_Rotation_Error_deg = percentile_simple(rotation_error_deg_list, 95);
        row.Mean_Normal_Angle_Error_deg = mean(normal_angle_error_deg);
        row.P95_Normal_Angle_Error_deg = percentile_simple(normal_angle_error_deg, 95);
        row.Mean_Abs_d_Error_um = mean(abs_d_error_um);
        row.P95_Abs_d_Error_um = percentile_simple(abs_d_error_um, 95);
        row.Mean_Raw_Normal_Orthogonality_Error = mean(raw_normal_orthogonality);
        row.OriginError_to_MeasuredPointRMS_Ratio = row.Mean_Origin_Error_mm / row.Measured_Point_RMS_mm;

        summary_rows = [summary_rows; struct2table(row)];

        raw_results(row_id).case_name = case_name;
        raw_results(row_id).distance_m = scene_distance_m;
        raw_results(row_id).origin_error_mm = origin_error_mm;
        raw_results(row_id).rotation_error_deg = rotation_error_deg_list;
        raw_results(row_id).normal_angle_error_deg = normal_angle_error_deg;
        raw_results(row_id).abs_d_error_um = abs_d_error_um;
        raw_results(row_id).measured_point_rms_mm = measured_point_rms_mm;
        raw_results(row_id).measured_point_p95_mm = measured_point_p95_mm;

        fprintf('Finished case %s, distance %.1f m\n', case_name, scene_distance_m);
        fprintf('Mean measured point RMS = %.6f mm\n', row.Measured_Point_RMS_mm);
        fprintf('Mean origin error = %.6f mm, P95 = %.6f mm\n', row.Mean_Origin_Error_mm, row.P95_Origin_Error_mm);
        fprintf('Mean rotation error = %.8f deg, P95 = %.8f deg\n', row.Mean_Rotation_Error_deg, row.P95_Rotation_Error_deg);
    end
end

%% ============================================================
%  Display summary tables
% =============================================================

disp(' ');
disp('================ Instrument-Error Propagation Summary ================');
disp(summary_rows);

% Extract combined case for a compact distance-to-error interpretation table.
combined_idx = summary_rows.Case == "Combined_AllRandom";
combined_table = summary_rows(combined_idx, :);

disp(' ');
disp('================ Combined Actual Random Error Case ================');
disp(combined_table(:, { ...
    'Distance_m', ...
    'Equivalent_DoubleAxis_PointSigma_mm', ...
    'Measured_Point_RMS_mm', ...
    'Mean_Origin_Error_mm', ...
    'P95_Origin_Error_mm', ...
    'Mean_Rotation_Error_deg', ...
    'P95_Rotation_Error_deg', ...
    'OriginError_to_MeasuredPointRMS_Ratio'}));

% Save numerical results for later comparison.
save('instrument_error_propagation_known_planes_results.mat', ...
    'summary_rows', 'combined_table', 'raw_results', 'cfg', 'distance_list_m');

try
    writetable(summary_rows, 'instrument_error_propagation_known_planes_summary.csv');
    writetable(combined_table, 'instrument_error_propagation_known_planes_combined_case.csv');
catch
    warning('Could not write CSV tables.');
end

%% ============================================================
%  Plots
% =============================================================

figure;
hold on;
case_list = unique(summary_rows.Case, 'stable');
for c = 1:length(case_list)
    idx = summary_rows.Case == case_list(c);
    plot(summary_rows.Distance_m(idx), summary_rows.Mean_Origin_Error_mm(idx), '-o', 'LineWidth', 1.5);
end
grid on;
xlabel('Measurement distance / m');
ylabel('Mean origin-position error / mm');
title('Instrument error propagation: mean origin error versus distance');
legend(cellstr(case_list), 'Location', 'best');

figure;
hold on;
for c = 1:length(case_list)
    idx = summary_rows.Case == case_list(c);
    plot(summary_rows.Distance_m(idx), summary_rows.P95_Origin_Error_mm(idx), '-o', 'LineWidth', 1.5);
end
grid on;
xlabel('Measurement distance / m');
ylabel('P95 origin-position error / mm');
title('Instrument error propagation: P95 origin error versus distance');
legend(cellstr(case_list), 'Location', 'best');

figure;
hold on;
for c = 1:length(case_list)
    idx = summary_rows.Case == case_list(c);
    plot(summary_rows.Distance_m(idx), summary_rows.Mean_Rotation_Error_deg(idx), '-o', 'LineWidth', 1.5);
end
grid on;
xlabel('Measurement distance / m');
ylabel('Mean rotation error / degree');
title('Instrument error propagation: mean rotation error versus distance');
legend(cellstr(case_list), 'Location', 'best');

figure;
plot(combined_table.Measured_Point_RMS_mm, combined_table.Mean_Origin_Error_mm, 'o-', 'LineWidth', 1.5);
grid on;
xlabel('Measured point RMS error / mm');
ylabel('Mean origin-position error / mm');
title('Combined actual error: origin error versus measured point RMS');

%% ============================================================
%  Local functions
% =============================================================

function s = make_error_case(name, use_roughness, use_range, use_angle)
    s.name = name;
    s.use_roughness = use_roughness;
    s.use_range = use_range;
    s.use_angle = use_angle;
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

function Pm = apply_spherical_instrument_noise(P, sigma_r, sigma_az, sigma_el, use_range, use_angle)
    x = P(:,1);
    y = P(:,2);
    z = P(:,3);

    r = sqrt(x.^2 + y.^2 + z.^2);
    az = atan2(y, x);
    el = asin(z ./ r);

    if use_range
        r = r + sigma_r * randn(size(r));
    end

    if use_angle
        az = az + sigma_az * randn(size(az));
        el = el + sigma_el * randn(size(el));
    end

    Pm = zeros(size(P));
    Pm(:,1) = r .* cos(el) .* cos(az);
    Pm(:,2) = r .* cos(el) .* sin(az);
    Pm(:,3) = r .* sin(el);
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

function val = acosd_safe(x)
    x = max(min(x, 1), -1);
    val = acos(x) * 180 / pi;
end

function angle_min = min_pairwise_normal_angle_deg(nMat)
    a12 = acosd_safe(abs(dot(nMat(:,1), nMat(:,2))));
    a13 = acosd_safe(abs(dot(nMat(:,1), nMat(:,3))));
    a23 = acosd_safe(abs(dot(nMat(:,2), nMat(:,3))));
    angle_min = min([a12, a13, a23]);
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
