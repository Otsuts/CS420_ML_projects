import numpy as np
import pandas as pd



def getdata(x_name):
    
    x_path = '../data/twin_pairs_X_3years_samesex.csv'
    y_path = '../data/twin_pairs_T_3years_samesex.csv'

    x_csv = pd.read_csv(x_path)
    age_list = list(x_csv[x_name])
    y_csv = pd.read_csv(y_path)
    weight_list1 = list(y_csv['dbirwt_0'])
    weight_list_2 = list(y_csv['dbirwt_1'])
    weight_list = [(weight_list1[i]+weight_list_2[i]) /
                   2 for i in range(len(weight_list1))]
    print(len(age_list), len(weight_list))
    
    return age_list, weight_list


if __name__ == '__main__':
    getdata()
