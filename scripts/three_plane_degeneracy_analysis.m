clc;
clear;
close all;

rng(3);

%% ============================================================
%  Degeneracy analysis for three-plane LiDAR calibration
%  Purpose:
%  Study how calibration accuracy changes when two plane normals
%  become close to parallel.
% =============================================================

num_trials = 200;

angle_list_deg = [90, 60, 45, 30, 20, 15, 10, 7, 5, 3];

noise_sigma = 0.002;   % 2 mm point cloud noise

num_points_per_board = 1200;
board_size_1 = 1.0;
board_size_2 = 0.8;

O_W = [0.30; -0.20; 0.40];

%% LiDAR pose relative to world
yaw   = deg2rad(35);
pitch = deg2rad(-10);
roll  = deg2rad(8);

R_WL = eulzyx(yaw, pitch, roll);
t_WL = [1.20; -0.60; 0.80];

%% Storage
num_angles = length(angle_list_deg);

condition_number_all = zeros(num_angles, 1);

origin_error_all = zeros(num_trials, num_angles);
rotation_error_all = zeros(num_trials, num_angles);
mean_residual_all = zeros(num_trials, num_angles);

%% ============================================================
%  Main loop over plane-normal angle
% =============================================================

for aidx = 1:num_angles

    theta = deg2rad(angle_list_deg(aidx));

    % Plane 1 normal
    n1 = [1; 0; 0];

    % Plane 2 normal becomes closer to n1 as theta decreases
    n2 = [cos(theta); sin(theta); 0];

    % Plane 3 normal is kept non-coplanar with the first two
    n3 = normalize_vec([0.20; -0.30; 1.0]);

    nW = zeros(3,3);
    nW(:,1) = normalize_vec(n1);
    nW(:,2) = normalize_vec(n2);
    nW(:,3) = normalize_vec(n3);

    dW = zeros(3,1);
    for i = 1:3
        dW(i) = -nW(:,i)' * O_W;
    end

    % Geometry quality
    N = nW';
    condition_number_all(aidx) = cond(N);

    % True board-defined frame
    [O_W_check, R_WB] = frame_from_three_planes(nW, dW);

    O_L_true = R_WL' * (O_W_check - t_WL);
    R_LB_true = R_WL' * R_WB;

    % True plane parameters in LiDAR frame
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

            points_L{i} = P_L;
        end

        % Fit three planes
        nL_est = zeros(3,3);
        dL_est = zeros(3,1);
        residual_each = zeros(3,1);

        for i = 1:3
            [n_fit, d_fit] = fit_plane_svd(points_L{i});

            % Align normal sign with ground truth for simulation
            if dot(n_fit, nL_true(:,i)) < 0
                n_fit = -n_fit;
                d_fit = -d_fit;
            end

            nL_est(:,i) = n_fit;
            dL_est(i) = d_fit;

            residual = points_L{i} * n_fit + d_fit;
            residual_each(i) = mean(abs(residual)) * 1000;
        end

        mean_residual_all(trial, aidx) = mean(residual_each);

        % Estimate board-frame origin and axes in LiDAR frame
        [O_L_est, R_LB_est] = frame_from_three_planes(nL_est, dL_est);

        % Evaluate
        origin_error_all(trial, aidx) = norm(O_L_est - O_L_true) * 1000;

        R_err = R_LB_est' * R_LB_true;
        rotation_error_all(trial, aidx) = rotation_angle_deg(R_err);

    end

    fprintf('Finished angle %.1f deg, cond(N)=%.2f\n', ...
        angle_list_deg(aidx), condition_number_all(aidx));
end

%% ============================================================
%  Summary statistics
% =============================================================

mean_origin = mean(origin_error_all, 1);
std_origin  = std(origin_error_all, 0, 1);
p95_origin  = zeros(1, num_angles);

mean_rot = mean(rotation_error_all, 1);
std_rot  = std(rotation_error_all, 0, 1);
p95_rot  = zeros(1, num_angles);

mean_residual = mean(mean_residual_all, 1);

for aidx = 1:num_angles
    p95_origin(aidx) = percentile_simple(origin_error_all(:,aidx), 95);
    p95_rot(aidx) = percentile_simple(rotation_error_all(:,aidx), 95);
end

summary_table = table( ...
    angle_list_deg(:), ...
    condition_number_all(:), ...
    mean_origin(:), ...
    std_origin(:), ...
    p95_origin(:), ...
    mean_rot(:), ...
    std_rot(:), ...
    p95_rot(:), ...
    mean_residual(:), ...
    'VariableNames', { ...
    'Angle_Between_Normal_1_2_deg', ...
    'Condition_Number', ...
    'Mean_Origin_Error_mm', ...
    'Std_Origin_Error_mm', ...
    'P95_Origin_Error_mm', ...
    'Mean_Rotation_Error_deg', ...
    'Std_Rotation_Error_deg', ...
    'P95_Rotation_Error_deg', ...
    'Mean_Plane_Residual_mm'});

disp(' ');
disp('================ Degeneracy Analysis Summary ================');
disp(summary_table);

%% ============================================================
%  Plot 1: condition number
% =============================================================

figure;
plot(angle_list_deg, condition_number_all, '-o', 'LineWidth', 2);
grid on;
set(gca, 'XDir', 'reverse');
xlabel('Angle between normal 1 and normal 2 / degree');
ylabel('Condition number of plane-normal matrix');
title('Geometry degeneracy as two plane normals become close');

%% Plot 2: origin error versus angle
figure;
errorbar(angle_list_deg, mean_origin, std_origin, '-o', 'LineWidth', 2);
grid on;
set(gca, 'XDir', 'reverse');
xlabel('Angle between normal 1 and normal 2 / degree');
ylabel('Origin error / mm');
title('Origin calibration error under geometric degeneracy');

%% Plot 3: rotation error versus angle
figure;
errorbar(angle_list_deg, mean_rot, std_rot, '-o', 'LineWidth', 2);
grid on;
set(gca, 'XDir', 'reverse');
xlabel('Angle between normal 1 and normal 2 / degree');
ylabel('Rotation error / degree');
title('Coordinate-axis error under geometric degeneracy');

%% Plot 4: origin error versus condition number
figure;
loglog(condition_number_all, mean_origin, '-o', 'LineWidth', 2);
grid on;
xlabel('Condition number of plane-normal matrix');
ylabel('Mean origin error / mm');
title('Origin error versus geometry condition number');

%% Plot 5: rotation error versus condition number
figure;
loglog(condition_number_all, mean_rot, '-o', 'LineWidth', 2);
grid on;
xlabel('Condition number of plane-normal matrix');
ylabel('Mean rotation error / degree');
title('Rotation error versus geometry condition number');

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