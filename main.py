from queue import Queue
from multiprocessing import Process

import fashion
import app


def main():
    shared_request_queue = Queue()
    shared_response_map = {}

    args = (shared_request_queue, shared_response_map)

    fashion_proc = Process(target=fashion.main, args=args)
    app_proc = Process(target=app.main, args=args)

    fashion_proc.run()
    app_proc.run()


if __name__ == "__main__":
    main()
