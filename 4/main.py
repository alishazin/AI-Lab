import string
import random
import math

def get_input(prompt, allowed_input=None, strip=True, err_msg=None, condition=None, is_int=False):
    
    while True:
        print()
        
        input_ = input(prompt)

        if strip: 
            input_ = input_.strip()

        if is_int:
            try: 
                input_ = int(input_)
            except:
                print("It must be an integer")
                continue

        if condition:
            if condition(input_):
                break
            else:
                print(err_msg)
        else:    
            if input_ not in allowed_input:
                if err_msg:
                    print(err_msg)
                else:
                    print(f"Invalid input, input must be one of {allowed_input}")
                continue
            break
    
    return input_

s_lc = list(string.ascii_lowercase)
s_uc = list(string.ascii_uppercase)
s_d = list(string.digits)
s_p = list(string.punctuation)

random.shuffle(s_lc)
random.shuffle(s_uc)
random.shuffle(s_d)
random.shuffle(s_p)

accepted_parts = [s_lc, s_uc]
generated_password = []

size = get_input(
    "Enter the size of the password: ", 
    strip=True, 
    is_int=1,
    err_msg="Size must be atleast 8", 
    condition=lambda x: x >= 8
)

include_d = get_input("Enter 1 to include digits, 0 to exclude: ", strip=True, allowed_input=['0','1'])
include_p = get_input("Enter 1 to include special characters, 0 to exclude: ", strip=True, allowed_input=['0','1'])

if include_d == '1': 
    accepted_parts.append(s_d)

if include_p == '1': 
    accepted_parts.append(s_p)

part_length = size * ((100/len(accepted_parts)) / 100)

for i in range(len(accepted_parts)):

    for k in range(math.floor(part_length)):
        generated_password.append(accepted_parts[i][k])

if len(generated_password) < size:

    char_set = s_lc
    if include_p:
        char_set = s_p
    elif include_d:
        char_set = s_d

    random.shuffle(char_set)

    for k in range(size-len(generated_password)):
        generated_password.append(char_set[k])

random.shuffle(generated_password)

generated_password = "".join(generated_password)

print(f"\nGenerated Password: {generated_password}")