# 基于传统机器学习和神经网络构建的分类模型
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, RidgeCV, ElasticNetCV
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from analysis import *

plt.figure(figsize=(9, 9))
plt.suptitle('Results of the Models')
colors = ['blue', 'green', 'purple']

# 线性回归模型构建和训练
plt.subplot(3, 2, 1)
plt.plot(range(60), Y_valid[:60], c='red', label='real values', lw=0.5)
plt.title('Linear regression model')
for i in range(1, 4):
    model1 = Pipeline([('Poly', PolynomialFeatures()), ('Linear', LinearRegression())])
    model1.set_params(Poly__degree= i)
    model1.fit(X_train, Y_train)
    Y_pre1 = model1.predict(X_valid)
    Y_result1 = [round(Y_pre1[i]) for i in range(len(Y_pre1))]
    plt.subplot(3, 2, 1)
    plt.plot(range(60), Y_result1[:60], c=colors[i-1], label=f"{i}-degree values", lw=0.5)
    plt.legend(loc='upper left', fontsize='small')
    score1 = f1_score(Y_valid, Y_result1, average='micro')
    print(f'{i}阶线性回归模型的F1 score为：', score1)
print()

# Ridge回归模型建构与训练
plt.subplot(3, 2, 2)
plt.plot(range(60), Y_valid[:60], c='red', label='real values', lw=0.5)
plt.title('Ridge regression model')
for i in range(1, 3):
    model2 = Pipeline([('Poly', PolynomialFeatures()), ('Linear', RidgeCV(alphas=np.logspace(-4, 2, 20)))])
    model2.set_params(Poly__degree= i)
    model2.fit(X_train, Y_train)
    Y_pre2 = model2.predict(X_valid)
    Y_result2 = [round(Y_pre2[i]) for i in range(len(Y_pre2))]
    plt.subplot(3, 2, 2)
    plt.plot(range(60), Y_result2[:60], c=colors[i-1], label=f"{i}-degree values", lw=0.5)
    plt.legend(loc='upper left', fontsize='small')
    score2 = f1_score(Y_valid, Y_result2, average='micro')
    print(f'{i}阶Ridge回归模型的F1 score为：', score2)
print()

# ElasticNet回归模型构建和训练
plt.subplot(3, 2, 3)
plt.plot(range(60), Y_valid[:60], c='red', label='real values', lw=0.5)
plt.title('ElasticNet regression model')
model3 = Pipeline([('Poly', PolynomialFeatures()), ('Linear', ElasticNetCV(alphas=np.logspace(-4,2, 20), l1_ratio=np.linspace(0, 1, 5)[1:]))])
model3.set_params(Poly__degree=1)
model3.fit(X_train, Y_train)
Y_pre3 = model3.predict(X_valid)
Y_result3 = [round(Y_pre3[i]) for i in range(len(Y_pre3))]
plt.subplot(3, 2, 3)
plt.plot(range(60), Y_result1[:60], c=colors[0], label=f"predicted values", lw=0.5)
plt.legend(loc='upper left', fontsize='small')
score3 = f1_score(Y_valid, Y_result1, average='micro')
print('ElasticNet回归模型的F1 score为：', score3)
print()


# 基于ID3算法的决策树模型构建与训练
plt.subplot(3, 2, 4)
plt.plot(range(60), Y_valid[:60], c='red', label='real values', lw=0.5)
plt.title('Decision tree model based on ID3 algorithm')
for i in range(3, 6):
    model4 = tree.DecisionTreeClassifier(criterion='entropy')
    model4.set_params(max_depth= i)
    model4.fit(X_train, Y_train)
    Y_pre4 = model4.predict(X_valid)
    plt.subplot(3, 2, 4)
    plt.plot(range(60), Y_pre4[:60], c=colors[i-3], label=f"{i}-max-depth values", lw=0.5)
    plt.legend(loc='upper left', fontsize='small')
    score4 = f1_score(Y_valid, Y_pre4, average='micro')
    print(f'最大深度为{i}的基于ID3算法的决策树模型的F1 score为：', score4)
print()

# 基于CART算法的决策树模型构建与训练
plt.subplot(3, 2, 5)
plt.plot(range(60), Y_valid[:60], c='red', label='real values', lw=0.5)
plt.title('Decision tree model based on CART algorithm')
for i in range(3, 6):
    model5 = tree.DecisionTreeClassifier(criterion='gini')
    model5.set_params(max_depth= i)
    model5.fit(X_train, Y_train)
    Y_pre5 = model5.predict(X_valid)
    plt.subplot(3, 2, 5)
    plt.plot(range(60), Y_pre5[:60], c=colors[i-3], label=f"{i}-max-depth values", lw=0.5)
    plt.legend(loc='upper left', fontsize='small')
    score5 = f1_score(Y_valid, Y_pre5, average='micro')
    print(f'最大深度为{i}的基于CART算法的决策树模型的F1 score为：', score5)
print()

# 神经网络模型建构与训练
plt.subplot(3, 2, 6)
plt.plot(range(60), Y_valid[:60], c='red', label='real values', lw=0.5)
plt.title('Nerual network')
model6 = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000, activation='relu', random_state=17)
model6.fit(X_train,Y_train)
Y_pre6 = model6.predict(X_valid)
plt.subplot(3, 2, 6)
plt.plot(range(60), Y_pre6[:60], c=colors[0], label=f"predicted values", lw=0.5)
plt.legend(loc='upper left', fontsize='small')
score6 = f1_score(Y_valid, Y_pre6, average='micro')
print('神经网络模型的F1 score为：', score6)
print()

plt.show()