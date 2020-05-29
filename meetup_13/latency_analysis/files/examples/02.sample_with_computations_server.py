import time
from random import randrange
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT = 8000


def primes(n):
    if n <= 2:
        return []
    sieve = [True] * (n + 1)
    for x in range(3, int(n ** 0.5) + 1, 2):
        for y in range(3, (n // x) + 1, 2):
            sieve[(x * y)] = False

    return [ 2 ] + [ i for i in range(3, n, 2) if sieve[i] ]


class Handler(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        f = randrange(1000000)
        primes(f)
        if f < 100000:
            for i in range(10):
                primes(f)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_GET(self):
        self._set_headers()
        self.wfile.write(("%f" % time.time()).encode('utf-8'))


httpd = HTTPServer(("0.0.0.0", PORT), Handler)
print("serving at port", PORT)

httpd.serve_forever()
