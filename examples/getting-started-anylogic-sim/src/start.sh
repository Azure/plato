#!/usr/bin/bash
set -xe
memcached -p 11211 -u memcache -l 127.0.0.1 -d

python -u main.py --test-local
