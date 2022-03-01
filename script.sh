#echo "shell /bin/bash" > /root/.screenrc
screen -dmS faceauth
screen -S faceauth -X stuff 'python face_auth/server.py\n'
tail -f /dev/null