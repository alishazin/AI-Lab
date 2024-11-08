
import string
import random
import math

lc_arr = list(string.ascii_lowercase)
uc_arr = list(string.ascii_uppercase)
p_arr = list(string.punctuation)
d_arr = list(string.digits)

parts = [lc_arr, uc_arr, p_arr, d_arr]

random.shuffle(lc_arr)
random.shuffle(uc_arr)
random.shuffle(p_arr)
random.shuffle(d_arr)

password = []

size = int(input("Enter length: "))

part_length = math.floor(size * 1 / 4)

for part in parts:
    password.extend(part[0:part_length])

if len(password) < size:
    random.shuffle(p_arr)
    password.extend(p_arr[0:size-len(password)])

random.shuffle(password)

print("".join(password))