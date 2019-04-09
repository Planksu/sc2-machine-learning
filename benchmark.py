import os

os.system("py -3.6 -O -m cProfile -o profile_name.prof blankbot.py")
os.system("snakeviz profile_name.prof")
