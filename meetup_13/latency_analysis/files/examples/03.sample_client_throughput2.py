#requests at the server processing throughput rate

import requests
import time
from threading import Thread
from queue import Queue

link = "http://192.168.56.1:8000/"

out = open("03.sample_throughput2.csv", "a")
out.write("LocalProcessTimeDiffMs,LocalTimeDiffMs,LocalMonotonicDiffMs,RemoteDiffTimeMs,RemoteProcessingTime\n");


def make_req():
  local_start_pt = time.process_time()
  local_start_time = time.time()
  local_start_monotonic = time.monotonic()

  f = requests.get(link)
  pong = f.content.decode("utf-8")
  times = pong.split(" ")
  remote_time = float(times[0])
  remote_proc_time = float(times[1])

  local_end_pt = time.process_time()
  local_end_time = time.time()
  local_end_monotonic = time.monotonic()

  out.write("%.9f,%.9f,%.9f,%.9f,%.9f\n" % (
    (local_end_pt - local_start_pt) * 1000,
    (local_end_time - local_start_time) * 1000,
    (local_end_monotonic - local_start_monotonic) * 1000,
    (remote_time - local_start_time) * 1000,
    (remote_proc_time * 1000)))
  out.flush()

def worker():
  while True:
    item = q.get()
    make_req()
    q.task_done()

q = Queue()
for i in range(1):
     t = Thread(target=worker)
     t.daemon = True
     t.start()

for i in range(10000):
  for i in range(10):
    q.put(1)
  time.sleep(0.01)

q.join()

out.close()
