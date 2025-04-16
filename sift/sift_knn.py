from sift_preprocess import *
from sift_feature_bow import *
from sift_evaluation import *
from sift_config import *
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# 1. 加载并分割数据集
print("\n[1] 加载数据...")
dataset_path = "../Aerial_Landscapes"
(train_images, train_labels), (test_images, test_labels), classes = \
    load_and_split_dataset(dataset_path, test_size=TEST_SIZE, sample_ratio=0.5)

# 2. 提取 SIFT 特征
print("[2] 提取训练集特征...")
train_descriptors = extract_color_sift_features(train_images)

# 3. 构建视觉词表
print("[3] 构建视觉词表...")
kmeans = create_visual_vocabulary(train_descriptors)

# 4. 转换 BoW 特征
print("[4] 生成 BoW 特征...")
X_train = extract_bow_features(train_descriptors, kmeans)
X_test = extract_bow_features(extract_color_sift_features(test_images), kmeans)
y_train = np.array(train_labels)
y_test = np.array(test_labels)

# 5. 标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. 训练 KNN 分类器
print("[5] 训练 KNN...")
clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train, y_train)

# 7. 评估模型
print("[6] 评估模型...")
y_pred = clf.predict(X_test)
evaluate_model(y_test, y_pred, classes)