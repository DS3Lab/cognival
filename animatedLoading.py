import sys
import time

#animation to know when script is running
def animatedLoading():
	chars = "/-\|"
	for char in chars:
		sys.stdout.write('\r'+'loading...'+char)
		time.sleep(.1)
		sys.stdout.flush()
	return 0