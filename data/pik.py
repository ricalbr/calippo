import pickle

# Open file in read-binary mode
with open('./6x6V_cal_11-2025.pickle', 'rb') as file:
    data = pickle.load(file)

print(data)
