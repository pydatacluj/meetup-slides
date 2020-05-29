import time, socket, threading, queue
from random import randrange
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT = 8000

q = queue.Queue()

class Handler(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_GET(self):
        self._set_headers()
        while not q.empty(): time.sleep(0.001) #this is simulating the waiting to process time
        stime = time.time()
        q.put(1)
        time.sleep(0.01)
        q.get(True, 1e5)
        etime = time.time()
        self.wfile.write(("%f %f" % (etime, etime - stime)).encode('utf-8'))


addr = ('', PORT)
sock = socket.socket (socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind(addr)
sock.listen(5)

class Thread(threading.Thread):
    def __init__(self, i):
        threading.Thread.__init__(self)
        self.i = i
        self.daemon = True
        self.start()
    def run(self):
        httpd = HTTPServer(("0.0.0.0", PORT), Handler)
        httpd.socket = sock
        httpd.server_bind = self.server_close = lambda self: None
        httpd.serve_forever()

[ Thread(i) for i in range(10) ]
time.sleep(1e5)