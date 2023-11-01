import eventlet
from eventlet import wsgi
import socketio
import threading

from hr_chat import main


class Server():
    def __init__(self):
        self.sio = socketio.Server(cors_allowed_origins="*")
        self.app = socketio.WSGIApp(self.sio)
        self.clients = {}

        @self.sio.event
        def connect(sid, *args):
            print(f"[{sid}] Connected")

        @self.sio.event
        def client_connected(client_sid, data):
            self.clients[client_sid] = {}
            data["client_sid"] = client_sid
            threading.Thread(target=main, kwargs=data, daemon=True).start()
        
        @self.sio.event
        def input_connected(input_sid, client_sid):
            self.sio.enter_room(input_sid, client_sid)
            self.clients[client_sid]["input"] = input_sid
        
        @self.sio.event
        def output_connected(output_sid, client_sid):
            self.sio.enter_room(client_sid, output_sid)
            self.clients[client_sid]["output"] = output_sid

        @self.sio.event
        def input(client_sid, data):
            self.sio.emit("input", data, room=client_sid)

        @self.sio.event
        def output(output_sid, data):
            self.sio.emit("output", data, room=output_sid)

        @self.sio.event
        def disconnect(sid):
            try:
                print(f"[{sid}] Disconnected")
                client = self.clients[sid]
                self.sio.disconnect(client["input"])
                self.sio.disconnect(client["output"])
            except KeyError:
                pass
        
        print("Starting server...")
        wsgi.server(eventlet.listen(("", 3001)), self.app, log_output=False)

if __name__ == "__main__":
    Server()