import numpy as np

def interval_score(Yu, Yl, Y, alpha = 0.05):
    out = Yu - Yl
    out += (Y > Yu).astype(int) * 2/alpha * (Y- Yu)
    out += (Y < Yl).astype(int) * 2/alpha * (Yl - Y)
    
    accuracy = np.sum(out > (Yu - Yl))
    accuracy /= len(out)

    return np.mean(out), accuracy