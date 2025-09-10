#%%
import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

dataset_dir = "dungeon_images_colour80"  

def load_images_from_folders(dataset_dir, img_size=(80, 80)):
    images = []
    labels = []
    class_names = os.listdir(dataset_dir)  
    
    for class_name in class_names:
        class_path = os.path.join(dataset_dir, class_name)
        if os.path.isdir(class_path):  
            for filename in os.listdir(class_path):
                img_path = os.path.join(class_path, filename)
                img = Image.open(img_path).resize(img_size)  
                img_array = np.array(img)  
                images.append(img_array)
                labels.append(class_name)  

    return np.array(images), np.array(labels)

X, y = load_images_from_folders(dataset_dir)

# Flatten images (80x80x3 to 19200 features)
X_flat = X.reshape(X.shape[0], -1) / 255.0  

le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"Total images: {X.shape[0]}, Classes: {len(le.classes_)}")
# %%
unique_classes, counts = np.unique(y, return_counts=True)
print("Classes:", unique_classes)
print("Class Distribution:", dict(zip(unique_classes, counts)))# %%
# %%
indices = np.random.choice(len(X), 5, replace=False)

fig, axes = plt.subplots(1, 5, figsize=(12, 6))
for i, idx in enumerate(indices):
    img = X[idx]  # Get the image
    label = y[idx]  # Get the assigned label
    axes[i].imshow(img)
    axes[i].set_title(label)
    axes[i].axis('off')
plt.show()

# %%
print("Image Shape Before Flattening:", X.shape)  
print("Image Shape After Flattening:", X_flat.shape)  
# %%
RUNS = 5
MAX_N_NEIGHBORS = 50

X_train, X_test, y_train, y_test = train_test_split(X_flat, y_encoded, test_size=0.2, random_state=42)

#%%
KNN_accs_train = np.zeros([RUNS, MAX_N_NEIGHBORS], dtype=np.float32)
KNN_accs_val = np.zeros([RUNS, MAX_N_NEIGHBORS], dtype=np.float32)

MAX_N_NEIGHBORS_RANGE = range(1, MAX_N_NEIGHBORS+1)

for i in range(RUNS):

  data_train_CV, data_val_CV, labels_train_CV, labels_val_CV = train_test_split(
        X_train, y_train, test_size=0.2, random_state=i)

  for j in MAX_N_NEIGHBORS_RANGE:

    knn = KNeighborsClassifier(n_neighbors=j)
    knn.fit(data_train_CV, labels_train_CV)

    KNN_predictions_train = knn.predict(data_train_CV)
    KNN_accs_train[i, j-1] = accuracy_score(labels_train_CV, KNN_predictions_train)

    KNN_predictions_val = knn.predict(data_val_CV)
    KNN_accs_val[i, j-1] = accuracy_score(labels_val_CV, KNN_predictions_val)

plt.figure()
plt.plot(MAX_N_NEIGHBORS_RANGE, KNN_accs_train.mean(axis=0), label='Train', linestyle="dashed")
plt.plot(MAX_N_NEIGHBORS_RANGE, KNN_accs_val.mean(axis=0), label='Validation', linestyle="solid")
plt.xlabel('n_neighbors (k)', fontsize=14)
plt.ylabel('Classification Accuracy', fontsize=14)
plt.grid()
plt.legend(fontsize=14)
plt.show()

best_KNN_index = np.argmax(KNN_accs_val.mean(axis=0))
best_N_NEIGHBOR = MAX_N_NEIGHBORS_RANGE[best_KNN_index]
best_KNN_accs_val = KNN_accs_val.mean(axis=0)[best_KNN_index]

print(f"The best N:{best_N_NEIGHBOR}")
print(f"The best Accuracy:{best_KNN_accs_val}")
# %%
best_N_NEIGHBOR = 1
final_knn = KNeighborsClassifier(n_neighbors=best_N_NEIGHBOR)
final_knn.fit(X_train, y_train)

final_KNN_predictions = final_knn.predict(X_test)
final_KNN_accuracy = accuracy_score(y_test, final_KNN_predictions)

print(f"Final KNN acc with best N =({best_N_NEIGHBOR}): {final_KNN_accuracy:.4f}")
# %%
class_names = le.classes_  

KNN_cm = confusion_matrix(y_test, final_KNN_predictions, labels=np.unique(y_test))
KNN_macro_precision = precision_score(y_test, final_KNN_predictions, average='macro')
KNN_macro_recall = recall_score(y_test, final_KNN_predictions, average='macro')
KNN_accuracy = accuracy_score(y_test, final_KNN_predictions)

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.3)
cax = ax.matshow(KNN_cm, cmap='Blues')

ax.set_xticks(np.arange(len(class_names)))
ax.set_yticks(np.arange(len(class_names)))
ax.set_xticklabels(class_names, fontsize=14)
ax.set_yticklabels(class_names, fontsize=14)

ax.set_xlabel('Predicted Labels', fontsize=16)
ax.set_ylabel('True Labels', fontsize=16)

for (i, j), val in np.ndenumerate(KNN_cm):
    ax.text(j, i, f'{val}', ha='center', va='center', color='red', fontsize=18)

print(f"Accuracy: {KNN_accuracy:.4f}")
print(f"Precision: {KNN_macro_precision:.4f}")
print(f"Recall: {KNN_macro_recall:.4f}")

plt.legend()
plt.tight_layout()
plt.show()

