from multiprocessing import Queue, Process, cpu_count


"""
this class was put in utils as there isn't any other obvious place where it
belongs. this routine is a generic multiprocessing routine for taking a function
which has to be run over multiple inputs and parallelizes it ove all available cpus.

This is an example script from the Github repo https://github.com/brmather/pycurious

"""


class GenericMultiprocessing:

    def __init__(self):
        pass

    def _func_queue(self, func, q_in, q_out, *args, **kwargs):
        """
        Retrive processes from the queue
        """
        while True:
            pos, var = q_in.get()
            if pos is None:
                break

            res = func(var, *args, **kwargs)
            q_out.put((pos, res))
            print(
                "finished azimuthal position #",
                pos,
                "with rwp = ",
                res[2] * 100.0,
                "%",
            )
        return

    def parallelise_function(self, var, func, *args, **kwargs):
        """
        Split evaluations of func across processors
        """
        n = len(var)

        processes = []
        q_in = Queue(1)
        q_out = Queue()

        # get the maximum number of processes that will be started
        nprocs = cpu_count()
        print("# of cpu = ", nprocs, "running on all of them.")

        for i in range(nprocs):
            pass_args = [func, q_in, q_out]

            p = Process(
                target=self._func_queue, args=tuple(pass_args), kwargs=kwargs
            )

            processes.append(p)

        for p in processes:
            p.daemon = True
            p.start()

        # put items in the queue
        sent = [q_in.put((i, var[i])) for i in range(n)]
        [q_in.put((None, None)) for _ in range(nprocs)]

        # get the results
        results = [[] for i in range(n)]
        for i in range(len(sent)):
            index, res = q_out.get()
            results[index] = res

        # wait until each processor has finished
        [p.join() for p in processes]
        p.terminate()
        # reorder results
        return results
