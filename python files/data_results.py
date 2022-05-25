import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# scan3 = {"head": 7.78, "face": 1.26}
# scan6 = {"head": 6.22, "face": 1.03}
# scan7 = {"head": 12.24, "face": 1.04}
# scan9 = {"head": 13.61, "face": 1.88}
# scan10 = {"head": 9.06, "face": 1.42}
# scan12 = {"head": 11.50, "face": 1.75}
# scan16 = {"head": 18.67, "face": 1.42}
# scan20 = {"head": 9.22, "face": 1.58}
# scan15 = {"head": 7.77, "face": 1.91}
# scan22 = {"head": 10.39, "face": 1.53}

head_error = np.array([7.78, 6.22, 12.24, 13.61, 9.06, 11.50, 18.67, 9.22, 7.77, 10.39])
face_error = np.array([1.26, 1.03, 1.04, 1.88, 1.42, 1.75, 1.42, 1.58, 1.91, 1.53])

print("chamfer distance full head average (mm)", head_error.mean(),"+/-", head_error.std())
print("chamfer distance face average (mm)", face_error.mean(),"+/-", face_error.std())

df = pd.DataFrame({'Head': head_error, "Face": face_error})

# ax1 = sns.boxplot(data=df["Head"]).set(xlabel='Head', ylabel='chamfer distance (mm)')
ax2 = sns.boxplot(data=df["Face"]).set(xlabel='Face', ylabel='chamfer distance (mm)')

plt.show()

