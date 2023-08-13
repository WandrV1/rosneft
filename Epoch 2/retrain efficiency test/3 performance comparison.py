import pandas as pd
import matplotlib.pyplot as plt

df_pretrained = pd.read_csv('Epoch 2\\retrain efficiency test\\3 performance logs\\pretrained_results.csv')
df_new = pd.read_csv('Epoch 2\\retrain efficiency test\\3 performance logs\\new_results.csv')

valid_pretrained = list(df_pretrained['valid_soft_metric'].dropna())
valid_new = list(df_new['valid_soft_metric'].dropna())

plt.plot(valid_pretrained, label='Предобученная модель')
plt.plot(valid_new, label='Новая модель')

plt.rcParams["figure.figsize"] = (20, 20)

plt.xlabel('Эпоха')
plt.ylabel('Dice with area')
plt.legend()
plt.show()
