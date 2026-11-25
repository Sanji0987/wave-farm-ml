import sys, numpy as np, pandas as pd
f, out = sys.argv[1], sys.argv[2]
df = pd.read_csv(f); N = sum(c.startswith('Power') for c in df.columns)
i = df['Total_Power'].idxmax(); r = df.loc[i]
x = r[[f'X{k}' for k in range(1,N+1)]].values.astype(float); y = r[[f'Y{k}' for k in range(1,N+1)]].values.astype(float)
np.savetxt(out, np.vstack([x,y]), delimiter=',', fmt='%.17g')
d = np.hypot(x[:,None]-x[None,:], y[:,None]-y[None,:]); iu = np.triu_indices(N,1)
print('row', i, 'x[:5]', x[:5], 'y[:5]', y[:5], 'min dist %.3f max %.3f' % (d[iu].min(), d[iu].max()), 'range x %.1f-%.1f y %.1f-%.1f' % (x.min(), x.max(), y.min(), y.max()))
# iso bracket from qW rounding
iso_lo = (df['Total_Power']/(N*(df['qW']+0.005))).max(); iso_hi = (df['Total_Power']/(N*(df['qW']-0.005))).min()
print('isolated power bracket from qW rounding: [%.1f, %.1f]' % (iso_lo, iso_hi))
