
import numpy as np
import math


def allocate_mux_tran (num_of_col, remaining_array, full_array, result_col, filter_num, filled_array_prev):
  filled_array = full_array - remaining_array
  max_col = int(np.max(filled_array)) # eshtebahe
  #last_non_zero_index = int(np.argmax(result_col))
  if np.any(filled_array != filled_array_prev):
    if np.all(result_col ==-1):
      selected_col =  max_col - 1
      result_col[0, selected_col] = filter_num + 1
    #elif (last_non_zero_index < max_col - 1):
      #selected_col =  max_col - 1
      #result_col[0, selected_col] = filter_num + 1
    else:
      m = 0
      for i in range(filled_array.shape[0]):
        if num_of_col[i , 0]!= 0:
          m = max(m, int(filled_array[i, 0]))
          
      for j in range(m - 1, result_col.shape[1]):
        if result_col[0, j] == -1:
          selected_col = j
          result_col[0, selected_col] = filter_num + 1
          break
  #print(result_col)    
  return  result_col, filled_array