import random
import time
from pathlib import Path
from typing import List

import imageio.v3 as iio
import numpy as np
import tyro

import viser
import viser.transforms as vtf
from viser.extras.colmap import (
    read_cameras_binary,
    read_images_binary,
    read_points3d_binary,
)


def main():




    server = viser.ViserServer()
    server.scene.add_icosphere( 
        name="/hello_sphere",
        radius=0.5,
        color=(255, 0, 0),
        position=(0.0, 0.0, 0.0),
    )
       

    print("Open your browser to http://localhost:8080")
    print("Press Ctrl+C to exit")

    while True:
        time.sleep(10.0)


if __name__ == "__main__":
    main()        