#!/usr/bin/python
from BaseHTTPServer import BaseHTTPRequestHandler,HTTPServer
from smile_local import SmileLocal
import numpy as np
import base64
import cv2
import cgi
import urlparse
from config import *

PORT_NUMBER = 3001

class VisionHandler(BaseHTTPRequestHandler):
    local = None


    def img_from_b64(self, b64_str):
        b_img = base64.b64decode(b64_str)
        nparr = np.fromstring(b_img, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


    def get_fields(self):
        ctype, pdict = cgi.parse_header(self.headers.getheader('content-type'))
        if ctype == 'multipart/form-data':
            postvars = cgi.parse_multipart(self.rfile, pdict)
        elif ctype == 'application/x-www-form-urlencoded':
            length = int(self.headers.getheader('content-length'))
            postvars = cgi.parse_qs(self.rfile.read(length), keep_blank_values=1)
        else:
            postvars = {}
        return postvars


    def verify_img(self):
        fields = self.get_fields()
        img = self.img_from_b64(fields['image'][0])
        identity = self.local.process_frame(img)
        if identity is None or identity == 'No_Match':
            identity = NO_MATCH
        return identity

    def train(self):
        fields = self.get_fields()
        identity = fields['identity'][0]
        train_imgs = list(map(lambda x: self.img_from_b64(x), fields['images']))
        self.local.train_new_identity(identity, train_imgs)
        return True


    def do_POST(self):
        if self.path.startswith("/verify"):
            identity = self.verify_img()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(identity)

        if self.path.startswith("/train"):
            resp = self.train()
            self.send_response(200)
            self.end_headers()
            if resp:
                self.wfile.write("success")
            else:
                self.wfile.write("failure")


try:
    l = SmileLocal()
    l.initialize_and_test()

    VisionHandler.local = l
    server = HTTPServer(('0.0.0.0', PORT_NUMBER), VisionHandler)
    print 'Started httpserver on port ' , PORT_NUMBER
    server.serve_forever()
except KeyboardInterrupt:
    print '^C received, shutting down..'
    server.socket.close()
