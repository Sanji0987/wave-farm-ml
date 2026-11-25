% Climate tables exactly as in Main.m (copied logic), for sites 1 (Perth) and 4 (Sydney)
% usage: octave climate.m <simdir> <outdir>
args = argv(); simdir = args{1}; outdir = args{2}; ncdir = args{3};
addpath(simdir); pkg load netcdf;
names = {'Perth','Adelaide','Tasmania','Sydney'};
files = {'HCtimeseries_aus_4m_114.80E_33.50S.nc','HCtimeseries_aus_4m_136.62E_36.07S.nc','HCtimeseries_aus_4m_143.75E_43.00S.nc','HCtimeseries_aus_4m_152.50E_34.00S.nc'};
for location = [1 4]
  fname = fullfile(ncdir, files{location});
  time = ncread(fname,'time'); hs = ncread(fname,'hs'); fp = ncread(fname,'fp');
  tm0m1 = ncread(fname,'tm0m1'); dirWave = ncread(fname,'dir');
  nraw = numel(hs);
  n1 = length(find(time==-32767)); n2 = length(find(hs==-32767)); n3 = length(find(fp==-32767));
  n4 = length(find(tm0m1==-32767)); n5 = length(find(dirWave==-32767));
  nMax = max([n1, n2, n3, n4, n5]);
  time(1:nMax) = []; hs(1:nMax) = []; fp(1:nMax) = []; tm0m1(1:nMax) = []; dirWave(1:nMax) = [];
  dirWave(dirWave == 360) = 0;
  xbins_hs = 0.5:1:[ceil(max(hs))+0.5];
  xbins_tp = round(1/max(fp)):1:round(1/min(fp));
  xbins_wa = 0:15:360;
  [counts, edges, mid, loc] = histcn([hs 1./fp dirWave], xbins_hs, xbins_tp, xbins_wa);
  N_hist_wa = squeeze(sum(sum(counts, 1), 2)); [s, I] = sort(N_hist_wa, 'descend');
  n_wa = 0; ii = 0; while n_wa < 0.90; ii = ii + 1; n_wa = n_wa + s(ii)/sum(s); end; I_wa = sort(I(1:ii));
  N_hist_hs = squeeze(sum(sum(counts, 3), 2)); [s, I] = sort(N_hist_hs, 'descend');
  n_hs = 0; ii = 0; while n_hs < 0.90; ii = ii + 1; n_hs = n_hs + s(ii)/sum(s); end; I_hs = sort(I(1:ii));
  N_hist_tp = squeeze(sum(sum(counts, 1), 3)); [s, I] = sort(N_hist_tp, 'descend');
  n_tp = 0; ii = 0; while n_tp < 0.90; ii = ii + 1; n_tp = n_tp + s(ii)/sum(s); end; I_tp = sort(I(1:ii));
  N_hist = counts(I_hs, I_tp, I_wa);
  Hs_hist = mid{1}(I_hs); Tp_hist = mid{2}(I_tp); waveAngle_hist = mid{3}(I_wa);
  nvalid = numel(hs); nbinned = sum(counts(:)); nkept = sum(N_hist(:));
  printf('%s: raw %d, after fill-trim %d, in histogram %d, kept %d (%.4f of trimmed)\n', names{location}, nraw, nvalid, nbinned, nkept, nkept/nvalid);
  printf('  Hs mids: %s\n  Tp mids: %s\n  dir mids: %s\n', mat2str(Hs_hist), mat2str(Tp_hist), mat2str(waveAngle_hist));
  save('-v7', fullfile(outdir, ['climate_' names{location} '.mat']), 'N_hist', 'Hs_hist', 'Tp_hist', 'waveAngle_hist', 'hs', 'fp', 'tm0m1', 'dirWave', 'time', 'nraw', 'nkept', 'nvalid');
end
