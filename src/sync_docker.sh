#!/bin/bash

docker build -t ele548 .
docker stop ele548
docker rm ele548
docker run -it --name ele548 -v "$PWD":/workspace ele548
