import helpers

# Interface between the server/local helper and the neural network
class SmileVerifier(object):
    def __init__(self):
        self.nn = None

    def verify(self, img):
        helpers.save_image(img)
        # find and return identity
        return
