import json
import asyncio
import websockets
import base64

async def run():
    async with websockets.connect("ws://raspberrypi.local:1234/ws") as ws:
        await ws.send(json.dumps({"type": "frame"}))
        msg = await ws.recv()
        data = json.loads(msg)["frame"].split("base64,")[1]

        with open("calibration.jpg", "wb") as fh:
            fh.write(base64.b64decode(data))

asyncio.run(run())
