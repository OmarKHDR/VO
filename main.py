#!/usr/bin/env python3
from vo.visual_odometry import visual_odometry
from calibration.calibration_samples_gen import generate_calibration_samples
from calibration.calibrate import calibrate_from_samples
from calibration.trycamera import init_camera
import cv2
import sys
import signal

def interrupt_handler(signum, frame):
    global cap
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    print("Goodbye!")
    sys.exit(0)

signal.signal(signal.SIGINT, interrupt_handler)

while True:
	cap = None
	print("hints: you can edit checkerboard size and the camera uri in [[calibration_params.json]]")
	prog = input("enter c for calibration, v for visual odometry, q to quit: ")

	if prog == 'c':
		cap = init_camera()
		generate_calibration_samples(cap)
		calibrate_from_samples()
	elif prog == 'v':
		cap = init_camera()
		visual_odometry(cap)
	elif prog == 'q':
		exit(0)

