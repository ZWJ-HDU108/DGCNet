% Training DATA generation - Three channel Version (real part, imaginary part, phase)+hot coded label
%*************************************************************************%
clear all;
close all;
clc;
tic;
%*************************************************************************%
% Location to save the DATA
filename_ECM = fullfile('E:\DOA_estimation','ECM_k=2.h5');
filename_SCM = fullfile('E:\DOA_estimation','SCM_k=2.h5');
%*************************************************************************%
SNR_dB_vec = -10: 2: 10;  % SNR range: -10dB to 10dB，stride 2dB
SOURCE_K = 2;           % Number of sources: 2个
ULA_M = 16;             % Number of array elements: 16
Max_DOA = 60;           % Field angle range: ±60°
grid_res = 1;           % Angular resolution: 1°
d = 0.5;                % Array element spacing: Half wavelength
%*************************************************************************%
% The steering/response vector of the N-element ULA
ULA_steer_vec = @(x, N, d) exp(1j * 2 * pi * d * sin(deg2rad(x)) * (0: 1: N-1)).';
%*************************************************************************%
% The training sets
grids = -Max_DOA: grid_res: Max_DOA;   % Discrete angle mesh
ang_d = nchoosek(grids, SOURCE_K);   % All possible source angle combinations
S = length(SNR_dB_vec);              % SNR levels
G = size(ang_d, 1);                  % Training angle vs. quantity

% 3 channels: real part, imaginary part, phase
R_ECM = zeros(ULA_M, ULA_M, 3, G, S);  % Expected covariance matrix (ECM)
R_SCM = zeros(ULA_M, ULA_M, 3, G, S);  % Sampled covariance matrix (SCM)

%*************************************************************************%

parfor i = 1: S  % Parallel processing of different SNR levels
    SNR_dB = SNR_dB_vec(i);
    noise_power = 10 ^ (-SNR_dB / 10);      % Calculate noise power
    
    % Temporary variable: 3 channels (real part, imaginary part, phase)
    r_the = zeros(ULA_M, ULA_M, 3, G);  % Temporary expected covariance variable at a SNR level
    r_sam = zeros(ULA_M, ULA_M, 3, G);  % Temporary sampled covariance variable at a SNR level
    
    for j = 1: G  % Traverse all angle combinations
        SOURCE_angles = ang_d(j, :);
        A_ula = zeros(ULA_M, SOURCE_K);
        
        % Constructing array manifold matrix
        for k = 1: SOURCE_K
            A_ula(:, k) = ULA_steer_vec(SOURCE_angles(k), ULA_M, d);
        end
        
        %% Calculate the theoretical covariance matrix (ECM)
        % Assuming that the source power is 1, it is uncorrelated
        Ry_the = A_ula * diag(ones(SOURCE_K, 1)) * A_ula' + noise_power * eye(ULA_M);
        
        %% Calculate sampling covariance matrix (SCM)
        T = 400;  % Snapshots
       
        S_real = randn(SOURCE_K, T); 
        S_imag = randn(SOURCE_K, T); 
        S_signal = (S_real + 1j * S_imag) / sqrt(2);
        
        % Array receive signal
        X = A_ula * S_signal;
        
        % Noise
        Eta_real = randn(ULA_M, T);
        Eta_imag = randn(ULA_M, T);
        Eta = sqrt(noise_power / 2) * (Eta_real + 1j * Eta_imag);
        
        % Received signal with noise
        Y = X + Eta;
        
        % Sampling covariance matrix
        Ry_sam = (Y * Y') / T;
        
        %% Extract three channel data
        % Real part
        r_the(:, :, 1, j) = real(Ry_the);
        r_sam(:, :, 1, j) = real(Ry_sam);
        
        % Imag part
        r_the(:, :, 2, j) = imag(Ry_the);
        r_sam(:, :, 2, j) = imag(Ry_sam);
        
        % Phase part
        r_the(:, :, 3, j) = angle(Ry_the);
        r_sam(:, :, 3, j) = angle(Ry_sam);
    end
    
    fprintf('Processing completed SNR = %d dB (level %d/%d)\n', SNR_dB, i, S);
    R_ECM(:, :, :, :, i) = r_the;
    R_SCM(:, :, :, :, i) = r_sam;
end

%% Save data to HDF5 file
% The angles Ground Truth
angles = ang_d;

% Save theoretical covariance matrix (ECM)
fprintf('\nSave theoretical covariance matrix to: %s\n', filename_ECM);
if exist(filename_ECM, 'file')
    delete(filename_ECM);
end

h5create(filename_ECM, '/angles', size(angles));
h5write(filename_ECM, '/angles', angles);

h5create(filename_ECM, '/ECM', size(R_ECM));
h5write(filename_ECM, '/ECM', R_ECM);
h5disp(filename_ECM);

% Save sampling covariance matrix data (SCM)
fprintf('\nSave sampling covariance matrix to: %s\n', filename_SCM);
if exist(filename_SCM, 'file')
    delete(filename_SCM);
end

h5create(filename_SCM, '/angles', size(angles));
h5write(filename_SCM, '/angles', angles);

h5create(filename_SCM, '/SCM', size(R_SCM));
h5write(filename_SCM, '/SCM', R_SCM);
h5disp(filename_SCM);