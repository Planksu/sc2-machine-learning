import subprocess
import sys

packages = ['sc2', 'random', 'cv2', 'os', 'time', 'numpy', 'math', 'tensorflow', 'keras']
modules = []

for i in packages:
	try:
		modules.append(__import__(i))
		print("Succesfully imported {}".format(i))
	except ImportError as e:
		print("Error importing {}".format(i))
		install(i)


async def install(package):
	subprocess.call([sys.executable, "-m", "pip", "install", package])