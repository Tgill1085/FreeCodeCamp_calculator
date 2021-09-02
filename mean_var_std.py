import numpy as np


def calculate(list):
  #Check that the list is the right amount of numbers
  if(len(list) != 9):
    raise ValueError("List must contain nine numbers.")

  #Turn the list into an array to work with
  ls = np.asarray(list)

  #Reshape the 1 dimension into a 3x3
  rls = np.reshape(ls, (3,3))

  #Calculate along the rows, columns, and flat values. Included float dtype as the example showed the results including them, and convert them back into a list so the tests can understand the values, they should not still be an array form.
  rls_mean_a1 = rls.mean(axis=0, dtype=np.float64).tolist()
  rls_mean_a2 = rls.mean(axis=1, dtype=np.float64).tolist()
  rls_mean_flat = np.mean(rls, dtype=np.float64).tolist()

  rls_var_a1 = rls.var(axis=0, dtype=np.float64).tolist()
  rls_var_a2 = rls.var(axis=1, dtype=np.float64).tolist()
  rls_var_flat = np.var(rls, dtype=np.float64).tolist()

  rls_std_a1 = rls.std(axis=0, dtype=np.float64).tolist()
  rls_std_a2 = rls.std(axis=1, dtype=np.float64).tolist()
  rls_std_flat = np.std(rls, dtype=np.float64).tolist()

  #Learned that max and min don't understand the float, took the dtype off.
  rls_max_a1 = rls.max(axis=0).tolist()
  rls_max_a2 = rls.max(axis=1).tolist()
  rls_max_flat = np.max(rls).tolist()

  rls_min_a1 = rls.min(axis=0).tolist()
  rls_min_a2 = rls.min(axis=1).tolist()
  rls_min_flat = np.min(rls).tolist()

  rls_sum_a1 = rls.sum(axis=0).tolist()
  rls_sum_a2 = rls.sum(axis=1).tolist()
  rls_sum_flat = np.sum(rls).tolist()

  #Gather the calculations together to return as a dictionary
  calculations = {
  'mean': [rls_mean_a1, rls_mean_a2, rls_mean_flat],
  'variance': [rls_var_a1, rls_var_a2, rls_var_flat],
  'standard deviation': [rls_std_a1, rls_std_a2, rls_std_flat],
  'max': [rls_max_a1, rls_max_a2, rls_max_flat],
  'min': [rls_min_a1, rls_min_a2, rls_min_flat],
  'sum': [rls_sum_a1, rls_sum_a2, rls_sum_flat]
}

  return calculations
  print(calculations)
