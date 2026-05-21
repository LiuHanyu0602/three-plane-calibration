clc;
clear;
close all;

rng(2);

%% ============================================================
%  Monte Carlo simulation for three-plane LiDAR calibration
% =============================================================

num_trials = 200;

noise_levels_mm = [0, 0.5, 1, 2, 5, 10];
noise_levels_m = noise_levels_mm * 1e-3;

num_points_per_board = 1200;
board_size_1 = 1.0;
board_size_2 = 0.8;

%% ============================================================
%  Fixed ground-truth three-plane configuration
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

% True plane parameters in LiDAR frame
nL_true = zeros(3,3);
dL_true = zeros(3,1);

for i = 1:3
    nL_true(:,i) = R_WL' * nW(:,i);
    dL_true(i) = nW(:,i)' * t_WL + dW(i);
end

%% ============================================================
%  Storage
% =============================================================

num_noise = length(noise_levels_m);

origin_error_all = zeros(num_trials, num_noise);
rotation_error_all = zeros(num_trials, num_noise);
axis_error_all = zeros(num_trials, num_noise, 3);
residual_all = zeros(num_trials, num_noise, 3);

%% ============================================================
%  Monte Carlo loop
% =============================================================

for s = 1:num_noise

    noise_sigma = noise_levels_m(s);

    for trial = 1:num_trials

        points_L = cell(3,1);

        for i = 1:3

            tangent_shift = [0.30 * randn; 0.30 * randn];

            P_W = sample_points_on_plane( ...
                nW(:,i), O_W, tangent_shift, ...
                num_points_per_board, board_size_1, board_size_2);

            P_L = (R_WL' * (P_W' - t_WL))';

            P_L = P_L + noise_sigma * randn(size(P_L));

            points_L{i} = P_L;
        end

        % Fit planes
        nL_est = zeros(3,3);
        dL_est = zeros(3,1);

        for i = 1:3
            [n_fit, d_fit] = fit_plane_svd(points_L{i});

            % In simulation, align normal sign with ground truth.
            % In real experiment, this should be replaced by a fixed
            % normal-orientation rule.
            if dot(n_fit, nL_true(:,i)) < 0
                n_fit = -n_fit;
                d_fit = -d_fit;
            end

            nL_est(:,i) = n_fit;
            dL_est(i) = d_fit;

            residual = points_L{i} * n_fit + d_fit;
            residual_all(trial, s, i) = mean(abs(residual)) * 1000;
        end

        % Estimate coordinate frame
        [O_L_est, R_LB_est] = frame_from_three_planes(nL_est, dL_est);

        % Evaluate
        origin_error_all(trial, s) = norm(O_L_est - O_L_true) * 1000;

        R_err = R_LB_est' * R_LB_true;
        rotation_error_all(trial, s) = rotation_angle_deg(R_err);

        for k = 1:3
            axis_error_all(trial, s, k) = ...
                acosd_clamped(dot(R_LB_est(:,k), R_LB_true(:,k)));
        end
    end

    fprintf('Finished noise level %.1f mm\n', noise_levels_mm(s));
end

%% ============================================================
%  Summary statistics
% =============================================================

mean_origin = mean(origin_error_all, 1);
std_origin  = std(origin_error_all, 0, 1);
p95_origin  = zeros(1, num_noise);

mean_rot = mean(rotation_error_all, 1);
std_rot  = std(rotation_error_all, 0, 1);
p95_rot  = zeros(1, num_noise);

mean_residual = squeeze(mean(mean(residual_all, 1), 3));

for s = 1:num_noise
    p95_origin(s) = percentile_simple(origin_error_all(:,s), 95);
    p95_rot(s) = percentile_simple(rotation_error_all(:,s), 95);
end

summary_table = table( ...
    noise_levels_mm(:), ...
    mean_origin(:), ...
    std_origin(:), ...
    p95_origin(:), ...
    mean_rot(:), ...
    std_rot(:), ...
    p95_rot(:), ...
    mean_residual(:), ...
    'VariableNames', { ...
    'Noise_mm', ...
    'Mean_Origin_Error_mm', ...
    'Std_Origin_Error_mm', ...
    'P95_Origin_Error_mm', ...
    'Mean_Rotation_Error_deg', ...
    'Std_Rotation_Error_deg', ...
    'P95_Rotation_Error_deg', ...
    'Mean_Plane_Residual_mm'});

disp(' ');
disp('================ Monte Carlo Summary ================');
disp(summary_table);

%% ============================================================
%  Plot results
% =============================================================

figure;
errorbar(noise_levels_mm, mean_origin, std_origin, '-o', 'LineWidth', 2);
grid on;
xlabel('Point cloud noise standard deviation / mm');
ylabel('Origin error / mm');
title('Origin calibration error under different noise levels');

figure;
errorbar(noise_levels_mm, mean_rot, std_rot, '-o', 'LineWidth', 2);
grid on;
xlabel('Point cloud noise standard deviation / mm');
ylabel('Rotation error / degree');
title('Coordinate-axis calibration error under different noise levels');

figure;
plot(noise_levels_mm, p95_origin, '-o', 'LineWidth', 2);
grid on;
xlabel('Point cloud noise standard deviation / mm');
ylabel('95th percentile origin error / mm');
title('95th percentile origin error');

figure;
plot(noise_levels_mm, p95_rot, '-o', 'LineWidth', 2);
grid on;
xlabel('Point cloud noise standard deviation / mm');
ylabel('95th percentile rotation error / degree');
title('95th percentile rotation error');

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

function [O, R] = frame_from_three_planes(nMat, dVec)
    N = nMat';

    O = N \ (-dVec(:));

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

function angle_deg = rotation_angle_deg(R)
    value = (trace(R) - 1) / 2;
    value = max(min(value, 1), -1);
    angle_deg = acos(value) * 180 / pi;
end

function angle_deg = acosd_clamped(x)
    x = max(min(x, 1), -1);
    angle_deg = acos(x) * 180 / pi;
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