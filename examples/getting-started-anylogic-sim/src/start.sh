#!/usr/bin/bash
set -xe
memcached -p 11211 -u memcache -l 127.0.0.1 -d

gunicorn \
	--worker-class uvicorn.workers.UvicornWorker \
	--bind '127.0.0.1:8000' \
	platotk.baobab:app &

python -u main.py --test-local
