
		
import time
# Parallel code
from sklearn.externals.joblib import Parallel, delayed

from fun import slowinc



print(__name__)
#print("CPUs %d "%(multiprocessing.cpu_count()))
if __name__ == '__main__':
	inputs=range(10)
	time_start = time.clock()
	[slowinc(i) for i in inputs]  # this takes 10 seconds
	time_elapsed = (time.clock() - time_start)
	print("Sequntial %d"%(time_elapsed))

	time_start = time.clock()
	tasks=(delayed(slowinc)(i) for i in range(10))
	print(tasks)
	Parallel(n_jobs=3)(tasks)  # this takes 3 seconds
	time_elapsed = (time.clock() - time_start)
	print("Parallel %d"%(time_elapsed))


	# pool=mp.Pool(processes=3)
	# pool.map(slowinc, inputs)
	# time_elapsed = (time.clock() - time_start)
	# print("Parallel %d"%(time_elapsed))
	
	