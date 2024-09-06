# Find S Algorithm
import csv

def train(filename, res_col, ignore_cols, yes):
    
    h = None
    
    with open(filename, 'r') as file:
        
        reader = csv.DictReader(file)
        
        for line in reader:
            
            if line[res_col] == yes:
                
                if h == None:
                    h = {}
                    for key in line:
                        if key not in ignore_cols and key != res_col:
                            h[key] = line[key]
                            
                else:
                    for key in line:
                        if key not in ignore_cols and key != res_col and h[key] != line[key]:
                            h[key] = '?'
                            
    return h 
            
def print_result(result):
    for k in result:
        print(f"{k} : {result[k]}")

h_result = train("data.csv", "Poisonous", ["Example"], "Yes")
print_result(h_result)
            
print()
            
h_result = train("data2.csv", "Goes", [], "Yes")
print_result(h_result)