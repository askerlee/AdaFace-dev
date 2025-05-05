#!/bin/bash
N=$1    # Get the first argument passed to the script
fish -c "rsync2 172.20.117.176,172.20.117.112 -avKL --exclude='*-db/' --exclude='*png' --exclude='*/embeddings.pt' --info=progress2 ~/adaprompt/logs/ shaohua@SERVER:adaprompt/logs/"
# Schedule the script to re-run after $N minutes
echo "~/adaprompt/sync_every_n_min.sh $N >> ~/adaprompt/sync_every_n_min.log 2>&1" | at now + "$N" minutes
