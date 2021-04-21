import os
import zmq
import sys
import time
import zlib
import pickle
import numpy as np

class CodeReader():
    def __init__(self):
        self.code_path = '/mnt/data2/dataset/SDM_images/models_arm_nonarn_half/latent/'
        self.codes = os.listdir(self.code_path)
    
    def get_code(self):
        if len(self.codes) > 0:
            code_name = self.codes.pop()
            code = np.load(os.path.join(self.code_path, code_name))
            return code
        else:
            return None

if __name__ == '__main__':
    context = zmq.Context()
    print("Connecting to server...")
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://127.0.0.1:7720")
    reader = CodeReader()
    while True:
        input1 = input()
        if input1 == 'q':
            sys.exit()

        pickle_c = pickle.dumps(reader.get_code())
        compressed_c = zlib.compress(pickle_c)
        socket.send(compressed_c)

        compressed_result = socket.recv()
        pickle_result = zlib.decompress(compressed_result)
        result = pickle.loads(pickle_result)
        print("Received reply: ", len(result), result)