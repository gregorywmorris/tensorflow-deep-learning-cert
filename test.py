import random

list = []

for method in dir(random):
    list.append(method)

print(list)
