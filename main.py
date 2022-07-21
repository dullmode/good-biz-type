import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm

p_df = pd.read_csv('data/input/processor.csv')
p_df = p_df.groupby('year', group_keys=False).apply(lambda x: x.loc[x['MOS transistor count'].idxmax()])
w_df = pd.read_csv('data/input/worker2.csv')

#delete index and merge.
p_df.reset_index(drop = True, inplace = True)
w_df.reset_index(drop = True, inplace = True)
df = pd.merge(w_df, p_df, on="year", how="left")

objective_vars = ['agriculture', 'fisheries', 'mining', 'construction', 'manifacture', 'infrastructure', 'infomation', 'transport', 'retail', 'finance', 'real-estate', 'research', 'accommodations-service','personal-service', 'education', 'medical','compound','etc','gov']

#save graph.
fig, ax1 = plt.subplots(1,1,figsize=(10,8))
ax2 = ax1.twinx()
ax1.bar(df.index,df["MOS transistor count"], color="lightblue")
ax2.plot(df[objective_vars], linestyle="solid")
handler1, label1 = ax1.get_legend_handles_labels()
handler2, label2 = ax2.get_legend_handles_labels()
ax1.legend(handler1+handler2,label1+label2,borderaxespad=0)
ax1.grid(True)
ax1.set_xticklabels(df['year'])
fig.savefig('data/output/graph/graph.png')

#save heatmap
objective_vars.append('MOS transistor count')
fig = sns.heatmap(df[l].corr(), annot=True, cmap='Blues')
plt.gcf().set_size_inches(15, 8)
fig.figure.savefig('data/output/graph/heatmap.png')
objective_vars.remove('MOS transistor count')

#simple regression analysis
for var in objective_vars:
    x = df.loc[:, 'MOS transistor count']
    y = df.loc[:, var]
    X = sm.add_constant(x)
    model = sm.OLS(y, X)
    results = model.fit()
    a = results.params[0]
    b = results.params[1]
    plt.plot(x, y,"o")
    plt.plot(x, a+b*x)

    title = f"const: {results.params[0]}, R-squared: {results.rsquared}"
    plt.title(title)

    plt.savefig(f'data/output/sra/{var}.png') 
    plt.clf()
