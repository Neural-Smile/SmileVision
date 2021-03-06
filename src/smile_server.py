#!/usr/bin/python
from BaseHTTPServer import BaseHTTPRequestHandler,HTTPServer
from preprocessor import Preprocessor
from model import Model
import numpy as np
import base64
import cv2
import cgi
import urlparse

PORT_NUMBER = 3001

# This server parses base64 encoded png images and runs them
# through the preprocessor and the verifier and should eventually
# be able to recognize identities
class VisionHandler(BaseHTTPRequestHandler):
    preprocessor = None
    model = None


    def img_from_b64(self, b64_str):
        b_img = base64.b64decode(b64_str)
        nparr = np.fromstring(b_img, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


    def get_fields(self):
        return urlparse.parse_qs(self.path.split('?')[1])


    def read_img(self, fields):
        b64_str= fields['image'][0]
        return self.img_from_b64(b64_str)


    def verify_img(self):
        fields = self.get_fields()
        img = self.read_img(fields)
        processed = self.preprocessor.process(img)
        ## to conform to same size/ratio across all inputs
        processed = self.preprocessor.resize(processed) 
        if len(processed) > 1:
            print("multiple faces")
        identity = self.model.verify(processed[0])
        return identity


    def train(self):
        fields = self.get_fields()
        img = self.read_img(fields)
        processed = self.preprocessor.process(img)
        ## to conform to same size/ratio across all inputs
        processed = self.preprocessor.resize(processed) 
        model.train(processed, fields['label'])
        return 200


    def do_POST(self):
        print("POST request")
        #TODO: need to parse endpoint, separate from params

        if self.path.startswith("/verify"):
            print("VERIFY request")
            identity = self.verify_img()
            self.send_response(200)
            self.wfile.write(identity)

        if self.path.startswith("/train"):
            print("TRAIN request")
            resp = self.train()
            self.send_response(resp)


try:
    VisionHandler.preprocessor = Preprocessor()
    model = Model(VisionHandler.preprocessor)
    model.initialize()
    VisionHandler.model = model
    server = HTTPServer(('0.0.0.0', PORT_NUMBER), VisionHandler)
    print 'Started httpserver on port ' , PORT_NUMBER
    server.serve_forever()
except KeyboardInterrupt:
    print '^C received, shutting down..'
    server.socket.close()
