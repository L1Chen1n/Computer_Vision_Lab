import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score, classification_report
import os
import pandas as pd
import seaborn as sns
import os


print(os.path.exists('Lab3/Chinese_MINST_Dataset/chinese_mnist.csv'))

csv_file = 'Lab3/Chinese_MINST_Dataset/chinese_mnist.csv'
df = pd.read_csv(csv_file)

img_dir = 'Lab3/Chinese_MINST_Dataset/data/data'

def load_image(file_name):
    image_path = os.path.join(img_dir, file_name)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (32, 32))
    return image.flatten()

X = []
y = []

for _, row in df.iterrows():
    image_name = f'input_{row['suite_id']}_{row['sample_id']}_{row['code']}.jpg'
    label = row['value']
    X.append(load_image(image_name))
    y.append(label)

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

knn = KNeighborsClassifier(n_neighbors=3)
Dt = DecisionTreeClassifier(random_state=42)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')  # Use 'weighted' for multi-class
recall = recall_score(y_test, y_pred, average='weighted')  # Use 'weighted' for multi-class
f1 = f1_score(y_test, y_pred, average='weighted')  # Use 'weighted' for multi-class

print(f'Accuracy: {accuracy * 100:.2f}%')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1-Score: {f1:.2f}')
print(classification_report(y_test, y_pred))

# for i in range(5):
#     plt.imshow(X_test[i].reshape(28, 28, 3))  # Reshape the flattened image
#     plt.title(f'Predicted: {y_pred[i]} | True: {y_test[i]}')
#     plt.show()

cm = confusion_matrix(y_test, y_pred)

# Step 2: Visualize confusion matrix using seaborn heatmap
# plt.figure(figsize=(6, 5))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# plt.xlabel('Predicted labels')
# plt.ylabel('True labels')
# plt.title('Confusion Matrix')
# plt.show()