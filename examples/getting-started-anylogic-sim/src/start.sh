#!/usr/bin/bash
set -xe
memcached -p 11211 -u memcache -l 127.0.0.1 -d

gunicorn \
	--worker-class uvicorn.workers.UvicornWorker \
	--bind '127.0.0.1:8000' \
	platotk.baobab:app &

export SIM_API_HOST=http://localhost:8000
export SIM_CONTEXT={}
export SIM_WORKSPACE=dummy
export SIM_ACCESS_KEY=dummy

N_SIM="$(find -type f -name '*_linux.sh' 2> /dev/null | wc -l)"
SIM_EXE="$(find -type f -name '*_linux.sh' 2> /dev/null)"
if [ $N_SIM -gt 1 ]
then
	echo "More than one sim found:"
	echo "$SIM_EXE"
	exit 1
fi

bash "$SIM_EXE" &
python -u main.py --test-local
