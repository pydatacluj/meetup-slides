import requests
import time

link = "http://192.168.56.1:8000/"

out = open("02.sample_with_computations_localhost.csv", "a")

out.write("LocalProcessTimeDiffMs,LocalTimeDiffMs,LocalMonotonicDiffMs,RemoteDiffTimeMs\n");

for i in range(100000):
  local_start_pt = time.process_time()
  local_start_time = time.time()
  local_start_monotonic = time.monotonic()

  f = requests.get(link)
  pong = f.content.decode("utf-8")
  remote_time = float(pong)

  local_end_pt = time.process_time()
  local_end_time = time.time()
  local_end_monotonic = time.monotonic()

  out.write("%.9f,%.9f,%.9f,%.9f\n" % (
    (local_end_pt - local_start_pt) * 1000,
    (local_end_time - local_start_time) * 1000,
    (local_end_monotonic - local_start_monotonic) * 1000,
    (remote_time - local_start_time) * 1000))
  out.flush()
  time.sleep(0.001)

out.close()