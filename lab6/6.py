import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.kernel_regression import KernelReg
np.random.seed(42)
X = np.linspace(0, 2*np.pi, 200)
y = np.sin(X) + 0.1 * np.random.randn(200)
model = KernelReg(y, X, 'c')
x_test = np.linspace(0, 2*np.pi, 200)
y_pred, _ = model.fit(x_test)
plt.figure(figsize=(10, 6))
plt.scatter(X, y,color='red',edgecolor='black',s=60,alpha=0.7,label="Training Data")
plt.plot(x_test,y_pred,color='blue',linewidth=3,label="LWR Fit")
plt.title("Locally Weighted Regression (LWR)")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend(frameon=True)
plt.tight_layout()
plt.show()
