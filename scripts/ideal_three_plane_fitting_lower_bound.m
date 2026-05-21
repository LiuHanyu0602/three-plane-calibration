clc;
clear;
close all;

rng(10);

%% ============================================================
%  Ideal Three-Plane Fitting Lower-Bound Experiment
%
%  Purpose:
%  This script isolates the best-case plane-fitting error propagation.
%  It does NOT use blind segmentation, RANSAC, or outlier rejection.
%  The point-to-plane labels are assumed known.
%
%  Fixed engineering setup:
%    - Workspace: approximately 5 m x 5 m
%    - Three planar boards: 1.0 m x 0.5 m
%    - Effective scanned points per board: 1200 > 1000
%    - Ideal boards with controllable normal roughness / point noise
%
%  It evaluates:
%    1. Plane normal angle error
%    2. Plane offset d error
%    3. Origin-position error
%    4. Rotation error
%    5. Axis orthogonality error
%    6. Empirical error transfer factor from point noise to calibration error
%
%  Recommended first run:
%    Use surface_roughness_sigma_mm = 0.01 mm = 10 um.
% ============================================================

%% ============================================================
%  Basic settings
% ============================================================

num_trials = 500;

workspace_size_m = 5.0;       % 5 m x 5 m engineering space
num_points_per_board = 1200;  % effective scanned points > 1000
board_size_1_m = 1.0;         % 1 m
board_size_2_m = 0.5;         % 0.5 m

% Noise / roughness scan, unit: mm.
% 0.01 mm = 10 um roughness / range precision lower-bound case.
noise_sigma_mm_list = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0];

% Instrument reference values for interpretation.
range_precision_um = 10;      % sigma_r = 10 um
turntable_repeatability_urad = 10;  % sigma_theta = 10 urad

% In this lower-bound experiment, we use known labels and direct SVD plane fitting.
% Noise model below is the board normal roughness model:
%     P_noisy = P_true + epsilon * n_true, epsilon ~ N(0, sigma^2)
% This isolates plane fitting precision.

%% ============================================================
%  Fixed three-plane geometry inside the 5 m space
% ============================================================

% Three non-degenerate plane normals. They are intentionally well-conditioned.
n_true = zeros(3,3);
n_true(:,1) = normalize_vec([1.0; 0.05; 0.02]);
n_true(:,2) = normalize_vec([0.03; 1.0; 0.08]);
n_true(:,3) = normalize_vec([0.06; 0.04; 1.0]);

% The true three-plane intersection point, used as the board-defined origin.
O_true = [0.30; -0.20; 0.40];  % m

% Plane offsets: n_i^T x + d_i = 0
d_true = zeros(3,1);
for i = 1:3
    d_true(i) = -n_true(:,i)' * O_true;
end

% Board center shifts within each plane, to create finite boards not exactly centered at the origin.
% These shifts are along each plane's local tangent basis and stay inside a compact 5 m workspace.
tangent_shift_m = [
     0.30,  0.10;
    -0.25,  0.20;
     0.15, -0.25];

% True board frame generated from the three true plane normals.
[O_check, R_true] = frame_from_three_planes(n_true, d_true);
geometry_condition_number = cond(n_true');
min_plane_angle_deg = min_pairwise_normal_angle_deg(n_true);

fprintf('\n================ Fixed Geometry ================\n');
fprintf('Workspace size: %.2f m x %.2f m\n', workspace_size_m, workspace_size_m);
fprintf('Board size: %.2f m x %.2f m\n', board_size_1_m, board_size_2_m);
fprintf('Points per board: %d\n', num_points_per_board);
fprintf('True origin: [%.4f, %.4f, %.4f] m\n', O_true(1), O_true(2), O_true(3));
fprintf('Check origin reconstruction from true planes: %.6e m\n', norm(O_check - O_true));
fprintf('Condition number kappa(N): %.4f\n', geometry_condition_number);
fprintf('Minimum pairwise normal angle: %.4f deg\n', min_plane_angle_deg);

%% ============================================================
%  Main noise-scan experiment
% ============================================================

all_summary = table();
all_trial_data = struct();

for s = 1:length(noise_sigma_mm_list)

    sigma_mm = noise_sigma_mm_list(s);
    sigma_m = sigma_mm / 1000.0;

    origin_error_mm = nan(num_trials,1);
    rotation_error_deg_list = nan(num_trials,1);
    mean_normal_angle_error_deg = nan(num_trials,1);
    max_normal_angle_error_deg = nan(num_trials,1);
    mean_abs_d_error_um = nan(num_trials,1);
    max_abs_d_error_um = nan(num_trials,1);
    axis_orthogonality_error = nan(num_trials,1);

    dx_mm = nan(num_trials,1);
    dy_mm = nan(num_trials,1);
    dz_mm = nan(num_trials,1);

    for trial = 1:num_trials

        n_est = zeros(3,3);
        d_est = zeros(3,1);
        normal_angle_errors = zeros(3,1);
        d_errors_um = zeros(3,1);

        for i = 1:3
            P_true = sample_points_on_plane( ...
                n_true(:,i), O_true, tangent_shift_m(i,:), ...
                num_points_per_board, board_size_1_m, board_size_2_m);

            % Ideal flat board with normal-direction roughness / point noise.
            % This is the lower-bound model: no RANSAC, no outlier, known board label.
            eps_normal = sigma_m * randn(num_points_per_board, 1);
            P_noisy = P_true + eps_normal * n_true(:,i)';

            [n_i, d_i] = fit_plane_svd(P_noisy);

            % Align normal sign with true normal.
            if dot(n_i, n_true(:,i)) < 0
                n_i = -n_i;
                d_i = -d_i;
            end

            n_est(:,i) = n_i;
            d_est(i) = d_i;

            normal_angle_errors(i) = angle_between_vectors_deg(n_i, n_true(:,i));
            d_errors_um(i) = abs(d_i - d_true(i)) * 1e6;  % m -> um
        end

        [O_est, R_est] = frame_from_three_planes(n_est, d_est);

        delta_O_mm = (O_est - O_true) * 1000.0;
        dx_mm(trial) = delta_O_mm(1);
        dy_mm(trial) = delta_O_mm(2);
        dz_mm(trial) = delta_O_mm(3);

        origin_error_mm(trial) = norm(delta_O_mm);
        rotation_error_deg_list(trial) = rotation_error_deg(R_est, R_true);
        mean_normal_angle_error_deg(trial) = mean(normal_angle_errors);
        max_normal_angle_error_deg(trial) = max(normal_angle_errors);
        mean_abs_d_error_um(trial) = mean(d_errors_um);
        max_abs_d_error_um(trial) = max(d_errors_um);
        axis_orthogonality_error(trial) = orthogonality_error(R_est);
    end

    % Summary for this sigma.
    summary = table();
    summary.Point_Noise_or_Roughness_mm = sigma_mm;
    summary.Mean_Origin_Error_mm = mean(origin_error_mm);
    summary.P95_Origin_Error_mm = percentile_simple(origin_error_mm, 95);
    summary.Std_Origin_Error_mm = std(origin_error_mm);
    summary.Mean_Rotation_Error_deg = mean(rotation_error_deg_list);
    summary.P95_Rotation_Error_deg = percentile_simple(rotation_error_deg_list, 95);
    summary.Mean_Normal_Angle_Error_deg = mean(mean_normal_angle_error_deg);
    summary.P95_Normal_Angle_Error_deg = percentile_simple(mean_normal_angle_error_deg, 95);
    summary.Mean_Abs_d_Error_um = mean(mean_abs_d_error_um);
    summary.P95_Abs_d_Error_um = percentile_simple(mean_abs_d_error_um, 95);
    summary.Mean_Axis_Orthogonality_Error = mean(axis_orthogonality_error);
    summary.Origin_Error_to_Point_Noise_Ratio = mean(origin_error_mm) / sigma_mm;
    summary.Rotation_Error_deg_per_mm_PointNoise = mean(rotation_error_deg_list) / sigma_mm;

    all_summary = [all_summary; summary]; %#ok<AGROW>

    % Save per-trial data.
    tag = sprintf('sigma_%g_mm', sigma_mm);
    tag = strrep(tag, '.', 'p');
    all_trial_data.(tag).dx_mm = dx_mm;
    all_trial_data.(tag).dy_mm = dy_mm;
    all_trial_data.(tag).dz_mm = dz_mm;
    all_trial_data.(tag).origin_error_mm = origin_error_mm;
    all_trial_data.(tag).rotation_error_deg = rotation_error_deg_list;
    all_trial_data.(tag).mean_normal_angle_error_deg = mean_normal_angle_error_deg;
    all_trial_data.(tag).mean_abs_d_error_um = mean_abs_d_error_um;

    fprintf('\nFinished sigma = %.4f mm\n', sigma_mm);
    fprintf('Mean origin error = %.6f mm, P95 = %.6f mm\n', ...
        summary.Mean_Origin_Error_mm, summary.P95_Origin_Error_mm);
    fprintf('Mean rotation error = %.8f deg, P95 = %.8f deg\n', ...
        summary.Mean_Rotation_Error_deg, summary.P95_Rotation_Error_deg);
    fprintf('Mean normal angle error = %.8f deg\n', summary.Mean_Normal_Angle_Error_deg);
    fprintf('Mean |d| error = %.6f um\n', summary.Mean_Abs_d_Error_um);
    fprintf('Origin error / point noise ratio = %.4f\n', summary.Origin_Error_to_Point_Noise_Ratio);
end

fprintf('\n================ Ideal Plane Fitting Lower-Bound Summary ================\n');
disp(all_summary);

%% ============================================================
%  Instrument interpretation table
% ============================================================

% Convert turntable repeatability to equivalent transverse point uncertainty.
distance_m = [5; 10; 50; 100; 150; 200];
sigma_r_mm = range_precision_um / 1000.0;  % um -> mm
sigma_theta_rad = turntable_repeatability_urad * 1e-6;

single_axis_transverse_mm = distance_m * sigma_theta_rad * 1000.0;
double_axis_equivalent_mm = sqrt(sigma_r_mm^2 + single_axis_transverse_mm.^2 + single_axis_transverse_mm.^2);

instrument_table = table( ...
    distance_m, ...
    single_axis_transverse_mm, ...
    double_axis_equivalent_mm, ...
    'VariableNames', { ...
    'Distance_m', ...
    'Single_Axis_Transverse_Error_mm', ...
    'Double_Axis_Equivalent_Point_Error_mm'});

fprintf('\n================ Instrument Error Interpretation ================\n');
fprintf('Range precision sigma_r = %.3f mm\n', sigma_r_mm);
fprintf('Turntable repeatability sigma_theta = %.3f urad\n', turntable_repeatability_urad);
disp(instrument_table);

%% ============================================================
%  Plots
% ============================================================

figure;
loglog(all_summary.Point_Noise_or_Roughness_mm, all_summary.Mean_Origin_Error_mm, 'o-', 'LineWidth', 1.5);
hold on;
loglog(all_summary.Point_Noise_or_Roughness_mm, all_summary.P95_Origin_Error_mm, 's-', 'LineWidth', 1.5);
grid on;
xlabel('Single-point normal roughness / point noise \sigma_p / mm');
ylabel('Origin-position error / mm');
title('Ideal known-label three-plane fitting: origin error vs point noise');
legend('Mean origin error', 'P95 origin error', 'Location', 'northwest');

figure;
loglog(all_summary.Point_Noise_or_Roughness_mm, all_summary.Mean_Rotation_Error_deg, 'o-', 'LineWidth', 1.5);
hold on;
loglog(all_summary.Point_Noise_or_Roughness_mm, all_summary.P95_Rotation_Error_deg, 's-', 'LineWidth', 1.5);
grid on;
xlabel('Single-point normal roughness / point noise \sigma_p / mm');
ylabel('Rotation error / degree');
title('Ideal known-label three-plane fitting: rotation error vs point noise');
legend('Mean rotation error', 'P95 rotation error', 'Location', 'northwest');

figure;
loglog(all_summary.Point_Noise_or_Roughness_mm, all_summary.Mean_Abs_d_Error_um, 'o-', 'LineWidth', 1.5);
hold on;
loglog(all_summary.Point_Noise_or_Roughness_mm, all_summary.P95_Abs_d_Error_um, 's-', 'LineWidth', 1.5);
grid on;
xlabel('Single-point normal roughness / point noise \sigma_p / mm');
ylabel('Plane offset |d| error / \mum');
title('Plane offset fitting error vs point noise');
legend('Mean |d| error', 'P95 |d| error', 'Location', 'northwest');

figure;
semilogx(all_summary.Point_Noise_or_Roughness_mm, all_summary.Origin_Error_to_Point_Noise_Ratio, 'o-', 'LineWidth', 1.5);
grid on;
xlabel('Single-point normal roughness / point noise \sigma_p / mm');
ylabel('Mean origin error / point noise ratio');
title('Noise averaging effect from plane fitting');

% Error ellipse for the 10 um case.
idx_10um = find(abs(all_summary.Point_Noise_or_Roughness_mm - 0.01) < 1e-12, 1);
if ~isempty(idx_10um)
    tag = 'sigma_0p01_mm';
    dx = all_trial_data.(tag).dx_mm;
    dy = all_trial_data.(tag).dy_mm;
    dz = all_trial_data.(tag).dz_mm;

    figure;
    plot_error_ellipse_2d(dx, dy, '\Delta x / mm', '\Delta y / mm', ...
        'Ideal fitting lower bound, 10 \mum roughness: \Delta x-\Delta y');

    figure;
    plot_error_ellipse_2d(dx, dz, '\Delta x / mm', '\Delta z / mm', ...
        'Ideal fitting lower bound, 10 \mum roughness: \Delta x-\Delta z');

    figure;
    plot_error_ellipse_2d(dy, dz, '\Delta y / mm', '\Delta z / mm', ...
        'Ideal fitting lower bound, 10 \mum roughness: \Delta y-\Delta z');
end

save('ideal_three_plane_fitting_lower_bound_results.mat', ...
    'all_summary', 'all_trial_data', 'instrument_table', ...
    'n_true', 'd_true', 'O_true', 'R_true', ...
    'geometry_condition_number', 'min_plane_angle_deg');

fprintf('\nSaved results to ideal_three_plane_fitting_lower_bound_results.mat\n');

%% ============================================================
%  Local functions
% ============================================================

function v = normalize_vec(v)
    v = v(:) / norm(v);
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

    e2_temp = n2 - dot(e1, n2) * e1;

    if norm(e2_temp) < 1e-12
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

function ang = angle_between_vectors_deg(a, b)
    a = a(:) / norm(a);
    b = b(:) / norm(b);
    value = abs(dot(a,b));
    value = max(min(value, 1), -1);
    ang = acos(value) * 180 / pi;
end

function minang = min_pairwise_normal_angle_deg(nMat)
    a12 = angle_between_vectors_deg(nMat(:,1), nMat(:,2));
    a13 = angle_between_vectors_deg(nMat(:,1), nMat(:,3));
    a23 = angle_between_vectors_deg(nMat(:,2), nMat(:,3));
    minang = min([a12, a13, a23]);
end

function e = orthogonality_error(R)
    % Frobenius norm of deviation from perfect orthogonality.
    e = norm(R' * R - eye(3), 'fro');
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

function plot_error_ellipse_2d(x, y, xlabel_name, ylabel_name, title_name)
    data = [x(:), y(:)];
    mu = mean(data, 1);
    Sigma = cov(data);
    chi2_val = 5.991;  % 95% for 2D Gaussian
    [V, D] = eig(Sigma);

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
    legend('Monte Carlo samples', '95% covariance ellipse', ...
        'Mean error center', 'True zero error', 'Location', 'best');
end
