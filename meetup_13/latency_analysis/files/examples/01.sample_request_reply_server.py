import time
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT = 8000

class Handler(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_GET(self):
        self._set_headers()
        self.wfile.write(("%f" % time.time()).encode('utf-8'))
        
httpd = HTTPServer(("0.0.0.0", PORT), Handler)
print("serving at port", PORT)
httpd.serve_forever()
