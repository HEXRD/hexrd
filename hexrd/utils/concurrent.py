import os


def distribute_tasks(num_tasks, max_workers=os.cpu_count()):
    # Distribute tasks as evenly as possible to each worker.
    # Returns a list of tuples, where each tuple in the list
    # represents the [start, stop) task range for a worker.
    result = []
    num_assigned = 0

    tasks_per_worker = num_tasks // max_workers
    remainder = num_tasks % max_workers
    for i in range(max_workers):
        if num_assigned == num_tasks:
            break

        tasks = tasks_per_worker
        if remainder != 0:
            tasks += 1
            remainder -= 1

        if i == 0:
            previous = 0
        else:
            previous = result[-1][1]

        result.append((previous, previous + tasks))
        num_assigned += tasks

    return result
