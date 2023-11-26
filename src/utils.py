import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix, accuracy_score


def adjusted_r_squared(r_squared: float, X: np.ndarray) -> float:
  """
  X: should not include the target variable
  """
  if len(X) <= 0:
      raise ValueError('Invalid input. Please provide a valid DataFrame.')
  n = len(X)
  k = len(X[0])
  adjusted_r_squared = 1 - ((1 - r_squared) * (n - 1) / (n - k - 1))
  return adjusted_r_squared


def plot_meshgrid(X_set, y_set, classifier, scalerX, step=0.25,
                  title: str = '', x1_label: str = '', x2_label: str = ''):
  """
  X_set: should be an unscaled 2-column matrix (-1, 2)
  y_set: should be an array of {0, 1}
  """
  X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 10, stop=X_set[:, 0].max() + 10, step=step),
                       np.arange(start=X_set[:, 1].min() - 1000, stop=X_set[:, 1].max() + 1000, step=step))
  X_flattened = np.array([X1.ravel(), X2.ravel()]).T
  predicted = classifier.predict(
      scalerX.transform(X_flattened)).reshape(X1.shape)

  plt.contourf(X1, X2, predicted,
               alpha=0.75, cmap=ListedColormap(('salmon', 'dodgerblue')))
  plt.xlim(X1.min(), X1.max())
  plt.ylim(X2.min(), X2.max())
  for i, j in enumerate(np.unique(y_set)):
      plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                  c=ListedColormap(('salmon', 'dodgerblue'))(i), label=j)

  plt.title(f'{title}')
  plt.xlabel(f'{x1_label}')
  plt.ylabel(f'{x2_label}')
  plt.legend()
  plt.show()
  return


def accuracy_and_confusion(actual, predicted):
  cm = confusion_matrix(actual, predicted)
  acsc = accuracy_score(actual, predicted)
  print(cm)
  print(acsc)
  return