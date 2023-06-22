# 数据分析与可视化
import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv('wine.csv')
train_dataset = dataset.head(4000)

def draw(factor: any, factor_name: str, i: int) -> None:
        """绘图模板函数"""
        plt.subplot(3, 4, i)
        plt.title(f'{factor_name}')
        plt.scatter(factor, train_dataset.quality, 5)
factors = [(train_dataset[idx], idx) for idx in train_dataset.columns[:-1]]

# 绘制葡萄酒品质与各因素间关系的图像
if __name__ == '__main__':
    plt.figure(figsize=(9, 9))
    for i in range(11): draw(factors[i][0], factors[i][1], i+1)
    plt.suptitle('Relationship between Wine Quality and Factors as Follows')
    plt.show()

# 根据相关性选取合适的评价标准
relationships = list(train_dataset.corr().quality.unique())[:-1]    
drop_cols = []
for i in range(len(relationships)):
    if abs(relationships[i]) < 0.05:
        drop_cols.append(train_dataset.columns[i])
if __name__ == '__main__':    
    for i in range(len(relationships)):
        print(f"{train_dataset.columns[i]}".ljust(20) + f"对葡萄酒品质的贡献程度是：{relationships[i]};")
    print("\n保留下来的因素有：")
    for i in (set(train_dataset.columns) - set(drop_cols)): print(i)
        
# 数据分类处理
dataset = dataset.drop(columns=drop_cols)
train_dataset = dataset.head(4000)
valid_dataset = dataset.tail(-4000)
X = dataset[dataset.columns]
Y = dataset['quality']
Y.ravel()
X_train = train_dataset[train_dataset.columns].drop(columns=['quality'])
Y_train = train_dataset['quality']
X_valid = valid_dataset[valid_dataset.columns].drop(columns=['quality'])
Y_valid = valid_dataset['quality']