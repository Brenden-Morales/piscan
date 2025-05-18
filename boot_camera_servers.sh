#!/usr/bin/env bash
# start_cameras.sh
# Must be run from inside an existing GNOME Terminal window.

if [ -z "$GNOME_TERMINAL_SERVICE" ]; then
  echo "⚠️  Please run this script from inside GNOME Terminal."
  exit 1
fi

gnome-terminal --tab --title="server" -- bash -ic 'pipenv run uvicorn ui_server:app --reload; exec bash'
gnome-terminal --tab --title="ui" -- bash -ic 'cd ./svelteApp && npm run dev -- --host --port 3333; exec bash'
gnome-terminal --tab --title="picam0" -- bash -ic "ssh -t picam@picam0.local 'python camera_server.py'; exec bash"
gnome-terminal --tab --title="picam1" -- bash -ic "ssh -t picam@picam1.local 'python camera_server.py'; exec bash"
gnome-terminal --tab --title="picam2" -- bash -ic "ssh -t picam@picam2.local 'python camera_server.py'; exec bash"
gnome-terminal --tab --title="picam3" -- bash -ic "ssh -t picam@picam3.local 'python camera_server.py'; exec bash"
gnome-terminal --tab --title="picam4" -- bash -ic "ssh -t picam@picam4.local 'python camera_server.py'; exec bash"
gnome-terminal --tab --title="picam5" -- bash -ic "ssh -t picam@picam5.local 'python camera_server.py'; exec bash"