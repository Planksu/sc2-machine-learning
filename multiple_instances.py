import sys
import subprocess

procs = []
for i in range(5):
    proc = subprocess.Popen([sys.executable, 'BlankBot.py', '{}in.csv'.format(i), '{}out.csv'.format(i)])
    procs.append(proc)

for proc in procs:
    proc.wait()