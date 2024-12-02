import string
import random
import math

lc_arr = list(string.ascii_lowercase)
uc_arr = list(string.ascii_uppercase)
d_arr = list(string.digits)
p_arr = list(string.punctuation)

parts = [lc_arr, uc_arr, d_arr, p_arr]

random.shuffle(lc_arr)
random.shuffle(uc_arr)
random.shuffle(d_arr)
random.shuffle(p_arr)

size = int(input("Enter the length of the password: "))

password = []
part_length = math.floor(size / 4)

for i in parts:
    password.extend(i[0:part_length])

if len(password) < size:
    random.shuffle(p_arr)
    password.extend(p_arr[0:size - len(password)])

random.shuffle(password)

print("Password: ", "".join(password))