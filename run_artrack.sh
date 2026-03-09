#!/bin/bash
xhost +local:docker
docker compose up -d
docker exec -it artrack_container bash
