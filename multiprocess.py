from multiprocessing import Process
from timeit import default_timer as timer
import time

def sleep_func(x, id):
        print(f'Sleeping for {x} sec for {id}')
        time.sleep(x)

if __name__ == '__main__':
        
        # initialize process objects
        proc1 = Process(target=sleep_func, args=[1, 1])
        proc2 = Process(target=sleep_func, args=[1, 2])
        proc3 = Process(target=sleep_func, args=[1, 3])

        
        # begin timer
        start = timer()
        
        # start processes
        proc1.start()
        proc2.start()
        proc3.start()
        
        # wait for both process to finish
        proc1.join()
        proc2.join()
        proc3.join()
        
        print('Time: ', timer() - start)