% Simulate many layouts in one Octave session (buoy parameters computed once; they don't depend on positions).
% usage: octave run_batch.m <simdir> <climate.mat> <layouts.csv> <k> <d> <kRcut> <out.csv>
%   layouts.csv: one layout per row, x1..xN then y1..yN (metres)
%   out.csv:     one row per layout: total power (W), q, seconds, then per-buoy power (W)
args = argv(); addpath(args{1});
global KR_CUTOFF; KR_CUTOFF = str2double(args{6});
C = load(args{2}); L = csvread(args{3});
kval = str2double(args{4}); dval = str2double(args{5});
N = size(L, 2) / 2;
siteOpts.waterDepth = 30; siteOpts.submergenceDepth = 3; siteOpts.waveFreqs = linspace(0.2, 2, 50);
siteOpts.location.siteName = 'x'; siteOpts.location.Hs = C.Hs_hist; siteOpts.location.Tp = C.Tp_hist;
siteOpts.location.waveAngle = C.waveAngle_hist*pi/180; siteOpts.location.N_hist = C.N_hist;
warning('off', 'all');
kPTO = kval*ones(N,3,50); dPTO = dval*ones(N,3,50);
buoy = set_buoy_parameter(kPTO, dPTO, N, siteOpts);
fid = fopen(args{7}, 'w');
for r = 1:size(L, 1)
  array.number = N; array.radius = 5*ones(1,N); array.sphereCoordinate = zeros(3,N);
  array.sphereCoordinate(1,:) = L(r, 1:N); array.sphereCoordinate(2,:) = L(r, N+1:end);
  array.sphereCoordinate(3,:) = -(siteOpts.submergenceDepth + array.radius);
  tic; [P, PB, q] = arrayBuoyPlacement(array, siteOpts, buoy, 0); t = toc;
  fprintf(fid, '%.6f,%.10f,%.1f', P, q, t); fprintf(fid, ',%.6f', PB); fprintf(fid, '\n'); fflush(fid);
  printf('layout %d/%d: %.1f W, q %.5f, %.0f s\n', r, size(L,1), P, q, t); fflush(stdout);
end
fclose(fid);
