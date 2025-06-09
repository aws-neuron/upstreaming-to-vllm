# SPDX-License-Identifier: Apache-2.0
import json
from typing import Any, Dict, Optional

import zmq


class Router:

    def __init__(self, bind_address: str = "tcp://*:5555"):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.ROUTER)
        self.socket.bind(bind_address)
        self.running = True

    def send_json(self, identity: bytes, data: Any) -> None:
        """Send JSON data to a specific client"""
        try:
            self.socket.send_multipart(
                [identity, b"", json.dumps(data).encode()])
        except (TypeError, ValueError) as e:
            print(f"JSON encode error: {e}")
            # Send error message
            error_msg = {"status": "error", "message": "JSON encode error"}
            self.socket.send_multipart(
                [identity, b"", json.dumps(error_msg).encode()])
        except zmq.ZMQError as e:
            print(f"ZMQ error while sending: {e}")
            raise

    def recv_json(self, timeout: Optional[int] = None) -> tuple[bytes, Dict]:
        """Receive JSON data with optional timeout"""
        try:
            if timeout is not None:
                poller = zmq.Poller()
                poller.register(self.socket, zmq.POLLIN)
                if not poller.poll(timeout):
                    raise TimeoutError("Receive timeout")

            identity, empty, message = self.socket.recv_multipart()
            return identity, json.loads(message.decode())

        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            # Send error response to client
            self.send_json(identity, {
                "status": "error",
                "message": "Invalid JSON format"
            })
            raise
        except zmq.ZMQError as e:
            print(f"ZMQ error while receiving: {e}")
            raise

    def cleanup(self):
        self.running = False
        self.socket.close()
        self.context.term()


class Dealer:

    def __init__(self,
                 identity: str,
                 connect_address: str = "tcp://localhost:5555"):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.DEALER)
        self.socket.setsockopt_string(zmq.IDENTITY, identity)
        self.socket.connect(connect_address)
        self.running = True

    def send_json(self, data: Any) -> None:
        """Send JSON data"""
        try:
            self.socket.send_multipart([b"", json.dumps(data).encode()])
        except (TypeError, ValueError) as e:
            print(f"JSON encode error: {e}")
            raise
        except zmq.ZMQError as e:
            print(f"ZMQ error while sending: {e}")
            raise

    def recv_json(self, timeout: Optional[int] = None) -> Dict:
        """Receive JSON data with optional timeout"""
        try:
            if timeout is not None:
                poller = zmq.Poller()
                poller.register(self.socket, zmq.POLLIN)
                if not poller.poll(timeout):
                    raise TimeoutError("Receive timeout")

            empty, message = self.socket.recv_multipart()
            return json.loads(message.decode())

        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            raise
        except zmq.ZMQError as e:
            print(f"ZMQ error while receiving: {e}")
            raise

    def cleanup(self):
        self.running = False
        self.socket.close()
        self.context.term()
