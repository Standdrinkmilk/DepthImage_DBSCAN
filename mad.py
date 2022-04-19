import numpy as np


def getMAD(s):
    median = np.median(s)
    print('median: {}'.format(median))
    # 这里的b为波动范围
    b = 1.4826
    mad = b * np.median(np.abs(s - median))
    print('mad: {}'.format(mad))

    # 确定一个值，用来排除异常值范围
    lower_limit = median - (3 * mad)
    upper_limit = median + (3 * mad)

    # print(mad, lower_limit, upper_limit)
    return lower_limit, upper_limit


number = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
lower_limit, upper_limit = getMAD(number)
print('lower_limit: {} upper_limit: {}'.format(lower_limit, upper_limit))
