import re
import socketio
import threading

from langroid.io.base import InputProvider, OutputProvider
from langroid.utils.constants import Colors


def input_processor(message):
    for _, color in vars(Colors()).items():
        message = message.replace(color, "")
    outputs = []
    color_split = message.split("][")
    for i, cur in enumerate(color_split):
        if i > 0:
            cur = "[" + cur
        if i < len(color_split) - 1:
            cur = cur + "]"
        match = re.search(r"\[[^\]]+\]", cur)
        if match:
            color = match.group()[1:-1]
        else:
            color = "black"
        for x in cur.split("\n"):
            outputs.append([color, re.sub(r"\[[^\]]+\]", "", x)])

    messages = []
    for o in range(len(outputs)):
        messages.append(outputs[o][1])

    return messages


class WebSocketInputProvider(InputProvider):
    def __init__(self, name, client_sid):
        super().__init__(name)
        self.returned_value = None
        self.sio = socketio.Client()
        self.client_sid = client_sid

        @self.sio.event
        def input(data):
            self.returned_value = data
        
        @self.sio.event
        def connect():
            self.sio.emit("input_connected", self.client_sid)
        
        self.sio.connect("http://0.0.0.0:3001")
        threading.Thread(target=self.setup, daemon=True).start()
    
    def setup(self):
        try:
            self.sio.wait()
        except KeyboardInterrupt:
            self.sio.disconnect()

    def __call__(self, message="", default=""):
        while self.returned_value is None:
            pass
        returned_value = self.returned_value
        self.returned_value = None
        return returned_value


class WebSocketOutputProvider(OutputProvider):
    def __init__(self, name: str, client_sid: str):
        super().__init__(name)
        self.streaming = False
        self.sio = socketio.Client()
        self.client_sid = client_sid

        @self.sio.event
        def connect():
            self.sio.emit("output_connected", self.client_sid)

        self.sio.connect("http://0.0.0.0:3001")
        threading.Thread(target=self.setup, daemon=True).start()
    
    def setup(self):
        try:
            self.sio.wait()
        except KeyboardInterrupt:
            self.sio.disconnect()

    def handle_message(self, message: str, streaming: bool = False):
        messages = input_processor(message)
        for m in messages:
            if (len(m) == 0):
                pass
            self.sio.emit("output", {"text": m, "streaming": streaming})

    def __call__(self, message: str, streaming: bool = False):
        if streaming:
            if self.streaming:
                self.handle_message(message, True)
            else:
                self.streaming = True
                self.handle_message(message)
        else:
            self.streaming = False
            self.handle_message(message)
