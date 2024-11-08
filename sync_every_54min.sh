#!/bin/bash
fish -c "rsync2 172.20.117.176,172.20.117.215 -avKL --exclude='*-db/' --exclude='*png' --exclude='*/embeddings.pt' --info=progress2 /home/lish/adaprompt/logs/ shaohua@SERVER:adaprompt/logs/"
echo "/home/lish/adaprompt/sync_every_54min.sh >> /home/lish/adaprompt/sync54min.log 2>&1" | at now + 54 minutes

