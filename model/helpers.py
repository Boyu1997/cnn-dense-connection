'''helper function'''
# make x divisible by y, used to satisfy the group convolution shape constraint
def make_divisible(x, y):
    return int((x // y + 1) * y) if x % y else int(x)
