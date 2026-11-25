import sys, numpy as np, pandas as pd
for f in sys.argv[1:]:
    df = pd.read_csv(f)
    N = sum(c.startswith('Power') for c in df.columns)
    iso = df['Total_Power']/(N*df['qW'])
    s = df[[f'Power{i}' for i in range(1,N+1)]].sum(axis=1)
    i = df['Total_Power'].idxmax()
    print(f.split('/')[-1], 'rows', len(df), 'N', N, 'iso median %.3f min %.3f max %.3f' % (iso.median(), iso.min(), iso.max()),
          '| max|sumPow-Total| %.3f' % (s-df['Total_Power']).abs().max(), '| best row', i, 'Total %.3f' % df.loc[i,'Total_Power'], 'qW %.6f' % df.loc[i,'qW'])
