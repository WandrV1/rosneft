import pandas as pd
import matplotlib.pyplot as plt

df_attribute = pd.read_csv('C:\\Users\\skril\\PycharmProjects\\rosneft\\Epoch 2\\6 attribute concept\\lightning_logs\\version_0\\metrics.csv')
df_classic = pd.read_csv('C:\\Users\\skril\\PycharmProjects\\rosneft\\Epoch 2\\additional cube model selection\\encoders_logs\\lightning_logs\\version_3\\metrics.csv')


attribute_valid_per_image_iou = list(df_attribute['valid_per_image_iou'].dropna())
attribute_valid_dataset_iou = list(df_attribute['valid_dataset_iou'].dropna())

classic_valid_per_image_iou = list(df_classic['valid_per_image_iou'].dropna())
classic_valid_dataset_iou = list(df_classic['valid_dataset_iou'].dropna())

# for item in attribute_valid_per_image_iou:
#     print(item)

# print()

# for item in classic_valid_per_image_iou:
#     print(item)


# plot attribute vs classic
# plt.plot(attribute_valid_per_image_iou, label='attribute_per_image_iou')
plt.plot(attribute_valid_dataset_iou, label='Модель с когерентностью')
# plt.plot(classic_valid_per_image_iou, label='classic_per_image_iou')
plt.plot(classic_valid_dataset_iou, label='Классическая модель')

# change plot size
plt.rcParams["figure.figsize"] = (20, 20)

plt.xlabel('Эпоха')
plt.ylabel('IoU')
plt.legend()
plt.show()
