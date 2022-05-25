import numpy as np

def check_time(time):
    

def split_time(time):
    nnegative = sum(time < 0)
    time_neg = time[:nnegative]
    time_pos = time[nnegative:]
    return
