# %%
Input = input("Please enter a test string: ")

if len(Input) < 6:
    print ("The string is too short.")
    print ("Please enter a string with at least 6 characters.")

# %%
2+2
# %%
Input = input("P1ease enter an integer: ")
number = int(Input) 

if number % 2 == 0:
    print ("The number is even.")
else:
    print ("The number is odd.")
# %%
a = int(input("Enter the first side of the triangle: "))
b = int(input("Enter the second side of the triangle: "))
c = int(input("Enter the third side of the triangle: "))

if a != b and b != c and a != c: 
    print("This is a scalene triangle.") 
elif a == b and b == c:
    print("This is an equilateral triangle.") 
else: 
    print("This is an isosceles triangle.") 

# %%
# writing a function
def f():
    pass
# )
# %%
dir()
# %%
import math
# %%
def volume (r):
    """Returns the volume of a sphere with radius r."""
    v = (4/3) * math.pi * r**3
    return v
# %%
volume(2)
# %%
def triangle_area(b,h):
    """Returns the area of a triangle with base b and height h."""
    return 0.5 * b * h      # 0.5 is a float
# %%
triangle_area(3,6)

# %%
def cm(feet=0, inches=0):
    """Converts a length from feet and inches to centimeters. Value 0
    represents the default value for keyword arguments feet and inches."""
    inches_to_cm = inches * 2.54
    feet_to_cm = feet * 12 * 2.54
    return inches_to_cm + feet_to_cm
# %%
feeet = cm(feet=5)
inchinchees = cm(inches=8)
print(feeet)

# %%
# creating and working with sets
thor = set()
dir(thor)
thor.add(43)

thor.discard(50)
# %%
len(thor)
thor.clear()
2 in thor

# %%
# working with lists
primes = [2, 3, 5, 7, 11, 13]
primes.append(17) 
primes.append(19) 
primes 

primes[0]
primes[-1]
# %%
# List example 
prime_numbers = [2, 3, 5, 7, 11, 13, 17] 
# Tuple example 
perfect_squares = (1, 4, 9, 16, 25, 36) 

print("List methods") 
print(dir (prime_numbers)) 
print(80 * "-")
print("Tuple methods ") 
print(dir (perfect_squares))
# %%
import sys
print(dir(sys))
# %%
import timeit
# %%
import logging
dir(logging)
# %%
import os
# %%
os.getcwd()
# %%
# os.chdir('') #change directory'
# %%
#logging and creating log file
LOG_FORMAT = "%(levelname)s %(asctime)s - %(message)s"
logging.basicConfig(filename='example.log', level=logging.DEBUG,
                     format=LOG_FORMAT,
                     filemode='w')
logger = logging.getLogger()
logger.info('Our first message.')
print(logger.level)
# %%
#Fibonaci sequence and memoization
import functools #for memoization in pip and python 3
@functools.lru_cache(maxsize = 1000)
def fibonacci(n):
    if n == 1:
        return 1
    elif n == 2:
        return 1
    elif n > 2:
        return fibonacci(n-1) + fibonacci(n-2)
for n in range(1, 51):
    print(n, ":", fibonacci(n))
# %%
#generating random numbers
import random
# %%
for i in range(10):
    print(random.random())
# %%
def my37_random():
    #Random, scale, shift, return...numbers between 3 and 7 
    return 4*random.random() + 3

for i in range(5):
    print(my37_random())
# %%
for i in range(5):
    print(random.uniform(3, 7))
# %%
for i in range(5):
    print(random.normalvariate(5, 0.5))
# %%
for i in range(5):
    print(random.randint(3, 7))
# %%
outcomes = ["rock", "paper", "scissors"]

for i in range(20):
    print(random.choice(outcomes))
# %%
# Reading CSV in python
path = "c:\\Users\\aladesuru\\Downloads\\google_stock_data.csv"
lines = [line for line in open(path)]
lines[0]
# %%
lines[1]
lines[0].strip().split(',')	
# %%
dataset = [line.strip().split(',')	for line in open(path)]
dataset[0]
# %%
import csv
# %%
from datetime import datetime
# %%
path = "c:\\Users\\aladesuru\\Downloads\\google_stock_data.csv"
file = open(path, newline='')
reader = csv.reader(file)
header = next(reader) # The first line is the header
print(header)

# %%
data = []
for row in reader:
    # row = [Date, Open, High, Low, Close, Volume, Adj. Close]
    date = datetime.strptime(row[0], '%m/%d/%Y')
    open_price = float(row[1]) # 'open' is a built-in function so can't be used
    high = float(row[2])
    low = float(row[3])
    close = float(row[4])
    volume = int(row[5]) # needs to be an integer not float
    adj_close = float(row[6])
    
    data.append([date, open_price, high, low, close, volume,
                 adj_close])
print(data[0])
# %%
# compute and store daily stock returns
returns_path = "c:\\Users\\aladesuru\\Downloads\\google_returns.csv"
file = open(returns_path, 'w')
writer = csv.writer(file)
writer.writerow(["Date", "Return"])

for i in range(len(data) - 1):
    todays_row = data[i]
    todays_date = todays_row[0]
    todays_price = todays_row[-1]
    yesterdays_row = data[i+1]
    yesterdays_price = yesterdays_row[-1]

    daily_return = (todays_price - yesterdays_price) / yesterdays_price
    formatted_date = todays_date.strftime('%m/%d/%Y')
    writer.writerow([formatted_date, daily_return])
# %%
import random

# %%
def random_walk(n):
    """Return coordinates after 'n' block random walk."""
    x = 0
    y = 0
    for i in range(10):
        step = random.choice(['N', 'S', 'E', 'W'])
        if step == 'N':
            y = y + 1
        elif step == 'S':
            y = y - 1
        elif step == 'E':
            x = x + 1
        else:
            x = x - 1
    return (x, y)

# %%
for i in range(25):
    walk = random_walk(10)
    print(walk, "Distance from home = ",
          abs(walk[0]) + abs(walk[1]))
# %%
# monte carlo simulation
def random_walk_2(n):
    """Return coordinates after 'n' block random walk."""
    x, y = 0, 0
    for i in range(n):
        (dx, dy) = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
        x += dx
        y += dy
    return (x, y)

#for i in range(25):    #walk = random_walk(10)    #print(walk, "Distance from home = ",          #abs(walk[0]) + abs(walk[1]))
# %%
number_of_walks = 10000

for walk_length in range(1, 31):
    no_transport = 0 # Number of walks 4 or fewer blocks from home
    for i in range(number_of_walks):
        (x, y) = random_walk_2(walk_length)
        distance = abs(x) + abs(y)
        if distance <= 5:
            no_transport += 1
    no_transport_percentage = float(no_transport) / number_of_walks
    print("walk size = ", walk_length,
          " / % of no transport = ", 100*no_transport_percentage)

# %%
#list comprehension
import numpy as np

# Set the seed value (any integer)
seed_value = 42
np.random.seed(seed_value)

# Generate random numbers
random_numbers = np.random.rand(5)  # For example, generate 5 random numbers between 0 and 1

print(random_numbers)

# %%
