#!/usr/bin/python
from BaseHTTPServer import BaseHTTPRequestHandler,HTTPServer
from smile_preprocessor import SmilePreprocessor
from smile_verifier import SmileVerifier
import numpy as np
import base64
import cv2
import cgi
import urlparse

PORT_NUMBER = 3001

# This server parses base64 encoded png images and runs them
# through the preprocessor and the verifier and should eventually
# be able to recognize identities
class SmileHttpHandler(BaseHTTPRequestHandler):
    preprocessor = None
    verifier = None

    def img_from_b64(self, b64_str):
        b_img = base64.b64decode(b64_str)
        nparr = np.fromstring(b_img, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    def verify_img(self):
        form_data = self.rfile.read(int(self.headers["Content-Length"]))
        fields = urlparse.parse_qs(form_data)
        b64_str= fields['image'][0]
        img = self.img_from_b64(b64_str)
        processed = self.preprocessor.process(img)
        if len(processed) > 1:
            print("multiple faces")
        identity = self.verifier.verify(processed[0])
        # return identity to user
        return

    def do_POST(self):
        if self.path == "/verify":
            return self.verify_img()

try:
    server = HTTPServer(('0.0.0.0', PORT_NUMBER), SmileHttpHandler)
    SmileHttpHandler.preprocessor = SmilePreprocessor()
    SmileHttpHandler.verifier = SmileVerifier()
    print 'Started httpserver on port ' , PORT_NUMBER
    server.serve_forever()
except KeyboardInterrupt:
    print '^C received, shutting down..'
    server.socket.close()
