from std.collection import List
from std.memory import UnsafePointer

struct Task:
    var fn_ptr: UnsafePointer[NoneType]
    var data: UnsafePointer[NoneType]

struct ThreadPool:
    var num_threads: Int
    var workers: List[ThreadWorker]
    var task_queue: List[Task]

    def __init__(out self, num_threads: Int):
        self.num_threads = num_threads
        self.workers = List[ThreadWorker]()
        self.task_queue = List[Task]()

    def __del__(deinit self):
        pass

    fn submit(ref self, task: Task):
        self.task_queue.append(task)

    fn wait_all(ref self):
        pass

    fn shutdown(ref self):
        self.workers.clear()

struct ThreadWorker:
    var id: Int

    def __init__(out self, id: Int):
        self.id = id