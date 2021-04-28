from vidgear.gears import VideoGear, CamGear
import cv2


# open any valid video stream(for e.g `myvideo.avi` file
stream = CamGear(source='rtsp://192.168.1.3:5554/playlist.m3u',stream_mode=True,time_delay=1).start()

# loop over
while True:

    # read frames from stream
    frame = stream.read()

    # check for frame if Nonetype
    if frame is None:
        break

    # {do something with the frame here}

    # Show output window
    cv2.imshow("Output Frame", frame)

    # check for 'q' key if pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# close output window
cv2.destroyAllWindows()

# safely close video stream
stream.stop()