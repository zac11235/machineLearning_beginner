#随机森林
from sklearn.model_selection import train_test_split

# 将数据集分成训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.ensemble import RandomForestClassifier

# 创建随机森林分类器
rfc = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
# 训练模型
rfc.fit(X_train, y_train)
# 预测结果
y_pred = rfc.predict(X_test)

# 计算准确率
accuracy = rfc.score(y_pred, y_test)
print("Accuracy:", accuracy)
