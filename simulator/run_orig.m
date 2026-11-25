% Run the ORIGINAL simulator functions (copy with settable kR cut-off) for given positions.
% usage: octave run_orig.m <simdir> <climate.mat> <pos.csv (2 x N)> <k> <d> <kRcut> <out.txt> [pto.mat mode]
args = argv(); addpath(args{1});
global KR_CUTOFF; KR_CUTOFF = str2double(args{6});
C = load(args{2}); Position = csvread(args{3});
N = size(Position,2); kval = str2double(args{4}); dval = str2double(args{5});
kPTO = kval*ones(N,3,50); dPTO = dval*ones(N,3,50);
if numel(args) >= 8
  P = load(args{8}); fn = fieldnames(P);
  printf('pto file fields: %s\n', strjoin(fn', ','));
end
array.number = N; array.radius = 5*ones(1,N); array.sphereCoordinate = zeros(3,N);
array.sphereCoordinate(1:2,:) = Position;
siteOpts.waterDepth = 30; siteOpts.submergenceDepth = 3; siteOpts.waveFreqs = linspace(0.2, 2, 50);
siteOpts.location.siteName = 'x'; siteOpts.location.Hs = C.Hs_hist; siteOpts.location.Tp = C.Tp_hist;
siteOpts.location.waveAngle = C.waveAngle_hist*pi/180; siteOpts.location.N_hist = C.N_hist;
array.sphereCoordinate(3,:) = -(siteOpts.submergenceDepth + array.radius);
warning('off','all');
tic; buoy = set_buoy_parameter(kPTO,dPTO,N,siteOpts); t1 = toc;
[ParrayW, ParrayBuoyW, qW] = arrayBuoyPlacement(array, siteOpts, buoy, 0); t2 = toc;
fid = fopen(args{7}, 'w');
fprintf(fid, 'total %.6f\nq %.10f\nsec_isolated %.2f\nsec_total %.2f\n', ParrayW, qW, t1, t2);
fprintf(fid, 'buoy %.6f\n', ParrayBuoyW);
fprintf(fid, 'Piso_w %.10e\n', buoy.PowerIsolated(1,:));
fclose(fid);
printf('N=%d total=%.3f q=%.8f time=%.1fs\n', N, ParrayW, qW, t2);
