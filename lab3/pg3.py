import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
pca = PCA(n_components=2)
data_reduced = pca.fit_transform(iris_df)
reduced_df = pd.DataFrame(data_reduced, columns=["PC1", "PC2"])
reduced_df['target'] = iris.target
plt.figure(figsize=(10, 15))
colors = ['r', 'g', 'b']
for i, target_value in enumerate(np.unique(iris.target)):
    plt.scatter(
        reduced_df[reduced_df['target'] == target_value]['PC1'],
        reduced_df[reduced_df['target'] == target_value]['PC2'],
        c=colors[i],
        label=iris.target_names[i]
    )
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA of iris dataset')
    plt.legend()
    plt.show()
