import os
import re
import socketio
import typer


class Client:
    def __init__(self, debug, nocache, local, rebuild):
        self.debug = debug
        self.nocache = nocache
        self.local = local
        self.rebuild = rebuild
        self.sio = socketio.Client()
        self.sent_message = False

        @self.sio.event
        def connect():
            self.sio.emit("client_connected", {"debug": self.debug, "nocache": self.nocache, "local": self.local, "rebuild": self.rebuild})

        @self.sio.event
        def output(data: str):
            if (data.startswith("<s>")):
                data = data.lstrip("<s>")
                streaming = True
            else:
                streaming = False

            match = re.search(r'^\[([^\]]+)\]', data)
            if (match):
              data = data[:match.start()] + data[match.end():]

            if (not streaming):
                data = "\n" + data

            print(data, end="")

        @self.sio.event
        def disconnect():
            os._exit(0)

        self.sio.connect("http://0.0.0.0:3001")

        try:
            while True:
                data = input()
                self.sio.emit("input", data)
        except KeyboardInterrupt:
           self.sio.disconnect()


app = typer.Typer()

@app.command()
def main(
    debug: bool = typer.Option(False, "--debug", "-d", help="debug mode"),
    nocache: bool = typer.Option(False, "--nocache", "-nc", help="don't use cache"),
    local: bool = typer.Option(False, "--local", "-l", help="use local model"),
    rebuild: bool = typer.Option(False, "--rebuild", "-r", help="rebuild the vector database")
) -> None:
      Client(debug, nocache, local, rebuild)

if __name__ == "__main__":
    app()