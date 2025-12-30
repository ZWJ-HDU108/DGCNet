% Training TL DATA generation 
%*************************************************************************%
clear all;
close all;
clc;
tic;
%*************************************************************************%
% Location to save the DATA
filename_ECM = fullfile('E:\DOA_estimation_database', 'ECM_k=4_coverage.h5');
filename_SCM = fullfile('E:\DOA_estimation_database', 'SCM_k=4_coverage.h5');
%*************************************************************************%
SNR_dB_vec = -10: 2: 10;  % SNR range
SOURCE_K = 4;           % Number of sources
ULA_M = 16;             % Number of array elements
Max_DOA = 60;           % Field angle range
grid_res = 1;           % Angular resolution
d = 0.5;                % Array element spacing

SAMPLES_PER_SNR = 50000;  % Number of samples per SNR level
MIN_SAMPLES_PER_ANGLE = 500;  % Make sure each angle appears at least this many times

%*************************************************************************%
% The steering/response vector
ULA_steer_vec = @(x, N, d) exp(1j * 2 * pi * d * sin(deg2rad(x)) * (0: 1: N-1)).';
%*************************************************************************%
grids = -Max_DOA: grid_res: Max_DOA;   
num_angles = length(grids);
all_ang_combinations = nchoosek(grids, SOURCE_K);
total_combinations = size(all_ang_combinations, 1);

fprintf('Angle points: %d\n', num_angles);
fprintf('All possible combinations: %d\n', total_combinations);

S = length(SNR_dB_vec);
rng(42);  % Set random seed

%% Defining intelligent sampling functions
% This function ensures that all angles are covered
fprintf('\nStart smart sampling...\n');

% Create angle to combination mapping
angle_to_combos = cell(num_angles, 1);
for angle_idx = 1: num_angles
    angle_val = grids(angle_idx);
    % Find all combinations that contain this angle
    contains_angle = any(all_ang_combinations == angle_val, 2);
    angle_to_combos{angle_idx} = find(contains_angle);
end

%% Generate sampling index for each SNR level
selected_indices_all = zeros(SAMPLES_PER_SNR, S);

for snr_idx = 1: S
    fprintf('\nProcess SNR level %d/%d (SNR = %d dB)...\n', snr_idx, S, SNR_dB_vec(snr_idx));
    
    % Initialize as column vector to ensure dimension consistency
    selected_indices = [];  
    angle_coverage = zeros(num_angles, 1);
    
    % Stage 1: ensure that each angle is fully covered
    fprintf('  Stage 1: ensure that each angle is fully covered...\n');
    max_iterations = SAMPLES_PER_SNR * 2;  % Prevent infinite loops
    iteration = 0;
    
    while min(angle_coverage) < MIN_SAMPLES_PER_ANGLE && ...
           length(selected_indices) < SAMPLES_PER_SNR && ...
           iteration < max_iterations
        
        iteration = iteration + 1;
        
        % Find the angle with the least coverage
        [min_cov, ~] = min(angle_coverage);
        under_covered = find(angle_coverage <= min_cov);
        
        % Randomly select an angle with insufficient coverage
        target_angle_idx = under_covered(randi(length(under_covered)));
        
        % Select randomly from the available combinations of this angle
        available_combos = angle_to_combos{target_angle_idx};
        available_combos = setdiff(available_combos, selected_indices);
        
        if ~isempty(available_combos)
            % Select a combination at random
            chosen_idx = available_combos(randi(length(available_combos)));
            selected_indices = [selected_indices; chosen_idx];  % Ensure column vector splicing
            
            % Update angle override count
            chosen_combo = all_ang_combinations(chosen_idx, :);
            for k = 1: SOURCE_K
                angle_val = chosen_combo(k);
                angle_idx = find(grids == angle_val);
                if ~isempty(angle_idx)
                    angle_coverage(angle_idx) = angle_coverage(angle_idx) + 1;
                end
            end
        end
        
        % Show progress
        if mod(length(selected_indices), 2000) == 0
            fprintf('%d combinations selected, minimum coverage: %d\n', ...
                length(selected_indices), min(angle_coverage));
        end
    end
    
    fprintf('Phase 1 completed: %d combinations selected\n', length(selected_indices));
    fprintf('Angle coverage: %d - %d\n', min(angle_coverage), max(angle_coverage));
    
    % Stage II: if the target quantity has not been reached, supplement at random
    if length(selected_indices) < SAMPLES_PER_SNR
        fprintf('  Stage 2: random replenishment to the target sample number...\n');
        
        remaining = setdiff(1:total_combinations, selected_indices);
        num_to_add = SAMPLES_PER_SNR - length(selected_indices);
        
        if num_to_add <= length(remaining)
            % Randomly select the supplement to ensure it is a column vector
            additional = remaining(randperm(length(remaining), num_to_add))';
        else
            % If not enough, repeat sampling is allowed
            additional = randi(total_combinations, num_to_add, 1);
        end
        
        % Ensure that they are column vectors before splicing
        selected_indices = [selected_indices(:); additional(:)];
    end
    
    % If the target quantity is exceeded, intercept
    if length(selected_indices) > SAMPLES_PER_SNR
        selected_indices = selected_indices(1: SAMPLES_PER_SNR);
    end
    
    % Disorganize order
    selected_indices = selected_indices(randperm(length(selected_indices)));
    
    % Storage (make sure it is a column vector)
    selected_indices_all(:, snr_idx) = selected_indices(:);
    
    % Verify the angular coverage of the current SNR
    final_coverage = zeros(num_angles, 1);
    for i = 1: SAMPLES_PER_SNR
        combo = all_ang_combinations(selected_indices_all(i, snr_idx), :);
        for k = 1: SOURCE_K
            angle_idx = find(grids == combo(k));
            if ~isempty(angle_idx)
                final_coverage(angle_idx) = final_coverage(angle_idx) + 1;
            end
        end
    end
    
    uncovered = sum(final_coverage == 0);
    fprintf('Final result - coverage:%d/%d angles (%.1f%%)\n', ...
        num_angles - uncovered, num_angles, ...
        (num_angles - uncovered)*100/num_angles);
    
    if uncovered > 0
        uncovered_angles = grids(final_coverage == 0);
        fprintf('Uncovered angle: %s\n', num2str(uncovered_angles));
    end
end

G = SAMPLES_PER_SNR;

%% Initialize data matrix
R_ECM = zeros(ULA_M, ULA_M, 3, G, S); 
R_SCM = zeros(ULA_M, ULA_M, 3, G, S); 
angles = zeros(G, SOURCE_K, S);       

%% Generate covariance matrix data
parfor i = 1: S 
    SNR_dB = SNR_dB_vec(i);
    noise_power = 10 ^ (-SNR_dB / 10);
    
    % Get the selected angle combination of the current SNR level
    current_indices = selected_indices_all(:, i);
    current_angles = all_ang_combinations(current_indices, :);
    
    r_the = zeros(ULA_M, ULA_M, 3, G);
    r_sam = zeros(ULA_M, ULA_M, 3, G);
    temp_angles = zeros(G, SOURCE_K);
    
    for ii = 1: G
        SOURCE_angles = current_angles(ii, :);
        temp_angles(ii, :) = SOURCE_angles;
        
        A_ula = zeros(ULA_M, SOURCE_K);
        for k = 1:SOURCE_K
            A_ula(:, k) = ULA_steer_vec(SOURCE_angles(k), ULA_M, d);
        end
        
        Ry_the = A_ula * eye(SOURCE_K) * A_ula' + noise_power * eye(ULA_M);

        T = 400;
        
        S_signal = (randn(SOURCE_K, T) + 1j * randn(SOURCE_K, T)) / sqrt(2);
        
        X = A_ula * S_signal;
        
        Eta = sqrt(noise_power/2) * (randn(ULA_M, T) + 1j * randn(ULA_M, T));
        
        Y = X + Eta;
        
        Ry_sam = (Y * Y') / T;
        
        r_the(:, :, 1, ii) = real(Ry_the);
        r_sam(:, :, 1, ii) = real(Ry_sam);
        
        r_the(:, :, 2, ii) = imag(Ry_the);
        r_sam(:, :, 2, ii) = imag(Ry_sam);
        
        r_the(:, :, 3, ii) = angle(Ry_the);
        r_sam(:, :, 3, ii) = angle(Ry_sam);
        
        % Show progress
        if mod(ii, 5000) == 0
            fprintf('SNR = %d dB: %d/%d samples processed\n', SNR_dB, ii, G);
        end
    end
    
    fprintf('Complete SNR=%d dB (level%d/%d)\n', SNR_dB, i, S);
    R_ECM(:, :, :, :, i) = r_the;
    R_SCM(:, :, :, :, i) = r_sam;
    angles(:, :, i) = temp_angles;
end

%% Reshape angle data
angles_reshaped = zeros(G*S, SOURCE_K);
for i = 1:S
    start_idx = (i-1)*G + 1;
    end_idx = i*G;
    angles_reshaped(start_idx:end_idx, :) = angles(:, :, i);
end

%% Final verification angle coverage
fprintf('\n=====================================\n');
fprintf('Final angle coverage statistics:\n');

angle_count = zeros(num_angles, 1);
for i = 1:size(angles_reshaped, 1)
    for j = 1:SOURCE_K
        angle_idx = find(grids == angles_reshaped(i, j), 1);
        if ~isempty(angle_idx)
            angle_count(angle_idx) = angle_count(angle_idx) + 1;
        end
    end
end

covered_angles = sum(angle_count > 0);
fprintf('Number of angles covered: %d/%d (%.1f%%)\n', covered_angles, num_angles, covered_angles*100/num_angles);
fprintf('Average occurrence per angle: %.1f\n', mean(angle_count));
fprintf('Least occurrence: %d\n', min(angle_count));
fprintf('Most occurrence: %d\n', max(angle_count));

uncovered = grids(angle_count == 0);
if ~isempty(uncovered)
    fprintf('Uncovered angle: %s\n', num2str(uncovered));
else
    fprintf('All angles covered!\n');
end

%% Save data to HDF5 file
fprintf('\nSave data...\n');

% Save ECM
if exist(filename_ECM, 'file')
    delete(filename_ECM); 
end
h5create(filename_ECM, '/angles', size(angles_reshaped));
h5write(filename_ECM, '/angles', angles_reshaped);
h5create(filename_ECM, '/ECM', size(R_ECM));
h5write(filename_ECM, '/ECM', R_ECM);
h5create(filename_ECM, '/SNR_dB', size(SNR_dB_vec));
h5write(filename_ECM, '/SNR_dB', SNR_dB_vec);
h5create(filename_ECM, '/samples_per_snr', [1, 1]);
h5write(filename_ECM, '/samples_per_snr', SAMPLES_PER_SNR);

fprintf('ECM data saved to: %s\n', filename_ECM);

% Save SCM
if exist(filename_SCM, 'file')
    delete(filename_SCM); 
end
h5create(filename_SCM, '/angles', size(angles_reshaped));
h5write(filename_SCM, '/angles', angles_reshaped);
h5create(filename_SCM, '/SCM', size(R_SCM));
h5write(filename_SCM, '/SCM', R_SCM);
h5create(filename_SCM, '/SNR_dB', size(SNR_dB_vec));
h5write(filename_SCM, '/SNR_dB', SNR_dB_vec);
h5create(filename_SCM, '/samples_per_snr', [1, 1]);
h5write(filename_SCM, '/samples_per_snr', SAMPLES_PER_SNR);

fprintf('SCM data save to: %s\n', filename_SCM);