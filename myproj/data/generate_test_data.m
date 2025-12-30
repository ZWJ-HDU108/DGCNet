%% DOA test data generator
clear; close all; clc;

%% system parameter setting
T = 400;                        % snapshots
SNR_dB = 0;                     % SNR (dB)
ULA_M = 16;                     % Number of array elements
d = 0.5;                        % Half wavelength array element spacing

%% Define test scenarios
% DOA_set_K_2_small = [30.2, 33.7];      % Two signal sources, small angle interval
DOA_set_K_2_large = [-23.5, 25.6];       % Two signal sources, large angle interval
DOA_set_K_1 = -13.8;                 % Single signal source
DOA_set_K_3 = [-39.1, -4.5, 27.8];       % Three signal sources
DOA_set_K_4 = [-49.4, -12.6, 10.7, 37.2];     % Four signal sources
DOA_set_K_5 = [-50.3, -20.5, 3.7, 23.5, 46.2];  % Five signal sources
DOA_set_K_6 = [-48.5, -21.4, -3.3, 9.9, 23.2, 46.6];  % Six signal sources
DOA_set_K_7 = [-48.5, -34.1, -14, -0.9, 10.9, 27.1, 44.1];  % Seven signal sources
DOA_set_K_8 = [-48.5, -34.1, -14, -0.9, 10.9, 27.1, 44.1, 56.7];  % Eight signal sources
DOA_set_K_9 = [-49.4, -34.6, -23.2, -13.8, 0.5, 11.3, 20.7, 36.9, 45.2];  % Nine signal sources
DOA_set_K_10 = [-49.5, -34.6, -23.1, -13.6, 0.5, 11.8, 20.3, 36.4, 45.9, 55.2];  % Ten signal sources


%% 生成测试数据并保存
% fprintf('Generate K2_small scenario test data...\n');
% [scm, angles] = generateSampleCovMatrix(DOA_set_K_2_small, T, SNR_dB, ULA_M, d);
% saveTestData('DOA_test_K_2_small_-5DB_100T.h5', scm, angles);
% 
% fprintf('Generate K2_large scenario test data...\n');
% [scm, angles] = generateSampleCovMatrix(DOA_set_K_2_large, T, SNR_dB, ULA_M, d);
% saveTestData('DOA_test_K_2_large_-5DB_100T.h5', scm, angles);
% 
% fprintf('Generate K1 scenario test data...\n');
% [scm, angles] = generateSampleCovMatrix(DOA_set_K_1, T, SNR_dB, ULA_M, d);
% saveTestData('DOA_test_K_1_-5DB_100T.h5', scm, angles);
% 
% fprintf('Generate K3 scenario test data...\n');
% [scm, angles] = generateSampleCovMatrix(DOA_set_K_3, T, SNR_dB, ULA_M, d);
% saveTestData('DOA_test_K_3_-5DB_100T.h5', scm, angles);
% 
% fprintf('Generate K4 scenario test data...\n');
% [scm, angles] = generateSampleCovMatrix(DOA_set_K_4, T, SNR_dB, ULA_M, d);
% saveTestData('DOA_test_K_4.h5', scm, angles);

% fprintf('Generate K5 scenario test data...\n');
% [scm, angles] = generateSampleCovMatrix(DOA_set_K_5, T, SNR_dB, ULA_M, d);
% saveTestData('DOA_test_K_5.h5', scm, angles);
% 
% fprintf('Generate K6 scenario test data...\n');
% [scm, angles] = generateSampleCovMatrix(DOA_set_K_6, T, SNR_dB, ULA_M, d);
% saveTestData('DOA_test_K_6.h5', scm, angles);

fprintf('Generate K7 scenario test data...\n');
[scm, angles] = generateSampleCovMatrix(DOA_set_K_7, T, SNR_dB, ULA_M, d);
saveTestData('DOA_test_K_7.h5', scm, angles);

fprintf('All test data generation completed！\n');

%% Generate sampling covariance matrix
function [scm, angles] = generateSampleCovMatrix(DOAs, T, SNR_dB, ULA_M, d)
    
    ULA_steer_vec = @(x, N, d) exp(1j * 2 * pi * d * sin(deg2rad(x)) * (0: 1: N-1)).';
    
    K = length(DOAs);
    A_ula = zeros(ULA_M, K);
    for k = 1: K
        A_ula(:, k) = ULA_steer_vec(DOAs(k), ULA_M, d);
    end
   
    SOURCE.power = ones(1, K).^2;
    noise_power = min(SOURCE.power) * 10^(-SNR_dB / 10);
    
    S = (randn(K, T) + 1j * randn(K, T)) / sqrt(2);
    
    X = A_ula * S;
    
    Eta = sqrt(noise_power) * (randn(ULA_M, T) + 1j * randn(ULA_M, T)) / sqrt(2);
   
    Y = X + Eta;
    
    Ry_sam = Y * Y' / T;
    
    scm(:, :, 1) = real(Ry_sam);
    scm(:, :, 2) = imag(Ry_sam);
    scm(:, :, 3) = angle(Ry_sam); 
    
    % Return angle (for recording, not as a label)
    angles(:) = DOAs';
end


function saveTestData(filename, scm, angles)
    % Delete existing files
    if exist(filename, 'file')
        delete(filename);
    end
    
 
    h5create(filename, '/SCM', size(scm));
    h5write(filename, '/SCM', scm);
    
    % Save angle information (only for recording real DOA)
    h5create(filename, '/angle', size(angles));
    h5write(filename, '/angle', angles);
    
    fprintf('Save to %s\n', filename);
end