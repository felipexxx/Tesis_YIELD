
from time import sleep
import os

def slowinc(x):
    sleep(1)  # take a bit of time to simulate real work
    print("Inside Function :: %d  ... Process %d"%(x, os.getpid()))
    return x + 1


