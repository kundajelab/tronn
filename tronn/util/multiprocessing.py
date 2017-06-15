"""Description: helpful functions for multiprocessing
"""

import multiprocessing
import Queue


def setup_multiprocessing_queue():
	"""Wrapper around multiprocessing so that only this script
	needs to import multiprocessing
	"""

	return multiprocessing.Queue()


def func_worker(queue):
    """Takes a tuple of (function, args) from queue and runs them

    Args:
      queue: multiprocessing Queue where each elem is (function, args)

    Returns:
      None
    """
    while not queue.empty():
        try:
            [func, args] = queue.get(timeout=0.1)
        except Queue.Empty:
            continue
        func(*args) # run the function with appropriate arguments
    
    return None


def run_in_parallel(queue, parallel=12):
    """Takes a filled queue and runs in parallel
    
    Args:
      queue: multiprocessing Queue where each elem is (function, args)
      parallel: how many to run in parallel

    Returns:
      None
    """
    pids = []
    for i in xrange(parallel):
        pid = os.fork()
        if pid == 0:
            func_worker(queue)
            os._exit(0)
        else:
            pids.append(pid)
            
    for pid in pids:
        os.waitpid(pid,0)
        
    return None

