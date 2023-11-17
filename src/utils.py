import numpy as np

def adjusted_r_squared(r_squared: float, X: np.ndarray) -> float:
  """
  :param X: should not include the target variable
  """
  if len(X) <= 0:
      raise ValueError('Invalid input. Please provide a valid DataFrame.')
  n = len(X)
  k = len(X[0])
  adjusted_r_squared = 1 - ((1 - r_squared) * (n - 1) / (n - k - 1))
  return adjusted_r_squared