@echo off
rem Publishes a Windows (DirectShow) webcam to the bundled RTSP server so the
rem pipeline container can read it. Keep this window open while demoing.
rem
rem Usage:  scripts\publish-webcam.bat            (uses the camera named "Camo")
rem         scripts\publish-webcam.bat "Integrated Camera"
rem List cameras:  ffmpeg -list_devices true -f dshow -i dummy
rem Needs: ffmpeg on PATH, and the mediamtx container running (--profile sim).
rem Pipeline must use SOURCE_URI=rtsp://mediamtx:8554/webcam

set CAM=%~1
if "%CAM%"=="" set CAM=Camo

echo Publishing "%CAM%" to rtsp://localhost:8554/webcam  (close this window to stop)
:loop
ffmpeg -hide_banner -loglevel warning -f dshow -framerate 30 -i video="%CAM%" ^
  -c:v libx264 -preset ultrafast -tune zerolatency -pix_fmt yuv420p -g 30 ^
  -rtsp_transport tcp -f rtsp rtsp://localhost:8554/webcam
echo ffmpeg exited, retrying in 3s...
timeout /t 3 /nobreak >nul
goto loop
