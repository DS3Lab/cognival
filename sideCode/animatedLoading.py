import sys
import time

#animation to know when script is running
def animatedLoading(completed, total):
	chars = "/-\|"
	for char in chars:
		sys.stdout.write('\r'+'loading...'+char)
		sys.stdout.write('\t'+str(completed)+"/"+str(total))
		time.sleep(.1)
		sys.stdout.flush()


# from __future__ import division
# import sys
#
# for i, _ in enumerate(p.imap_unordered(do_work, xrange(num_tasks)), 1):
#     sys.stderr.write('\rdone {0:%}'.format(i/num_tasks))