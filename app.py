from flask import Flask, jsonify
import uuid
from time import sleep


def main(request_queue, response_map: dict):
    app = Flask(__name__)

    @app.post("/api")
    async def post_image():
        id = uuid.uuid4()

        # save image

        request_queue.put(f"{id}.jpg")

        while response_map.get(id, None) is None:
            sleep(1)

        response = response_map[id]

        return jsonify({"status": "success", "msg": response})
