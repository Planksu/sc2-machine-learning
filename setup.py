import subprocess
import sys

modules = ['sc2', 'random', 'cv2', 'os', 'time', 'numpy', 'math', 'tensorflow', 'keras']
ready_modules = []

def install(package):
	subprocess.call([sys.executable, "-m", "pip", "install", package])

for i in modules:
	try:
		ready_modules.append(__import__(i))
		print("Succesfully imported {}".format(i))
	except ImportError as e:
		print("Error importing {}".format(i))
		if(i == 'cv2'):
			install('opencv-python')
		else:
			install(i)
