import sys
import cv2
import os
from sys import platform
import argparse
import time
import numpy as np
import joblib

try:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        # Windows Import
        if platform == "win32":
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append(dir_path + './python/openpose/Release');
            os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + './Release;' +  dir_path + './bin;'
            import pyopenpose as op
        else:
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append('./python');
            # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
            # sys.path.append('/usr/local/python')
            from openpose import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e

    # Load the trained model
    # model = joblib.load('MAsquat_modelv2.pkl')
    model = joblib.load('MAsquat_modelv2.pkl')

    # Flags
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_display", default=False, help="Enable to disable the visual display.")
    parser.add_argument("--video", default="", help="Path to video file. If not provided, the camera will be used.")
    parser.add_argument("--image", default="", help="Process an image.")
    args = parser.parse_known_args()

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "./models/"
    # Add others in path?
    for i in range(0, len(args[1])):
        curr_item = args[1][i]
        if i != len(args[1])-1:
            next_item = args[1][i+1]
        else:
            next_item = "1"
        if "--" in curr_item and "--" in next_item:
            key = curr_item.replace('-','')
            if key not in params:
                params[key] = "1"
        elif "--" in curr_item and "--" not in next_item:
            key = curr_item.replace('-','')
            if key not in params:
                params[key] = next_item

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    if args[0].image:
        # Process a single image
        image = cv2.imread(args[0].image)
        if image is None:
            print("Failed to load the image.")
            sys.exit(-1)

        datum = op.Datum()
        datum.cvInputData = image
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))

        keypoints = datum.poseKeypoints

        if keypoints is not None and keypoints.shape:
            required_indices = [12, 9, 13, 10, 14, 11, 19, 20, 22, 23]
            required_keypoints = keypoints[0, required_indices, :].reshape(1, -1)

            prediction = model.predict(required_keypoints)

            if prediction[0] == 0:
                result = "INVALID SQUAT"
            else:
                result = "VALID SQUAT"

            cv2.putText(datum.cvOutputData, result, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

            prediction_frame = np.zeros((200, 400, 3), dtype=np.uint8)
            cv2.putText(prediction_frame, result, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Prediction", prediction_frame)

            cv2.imshow("OpenPose - Squat Analysis", datum.cvOutputData)
            cv2.waitKey(0)

    else:
        # Start video capture
        if args[0].video:
            cap = cv2.VideoCapture(args[0].video)
        else:
            cap = cv2.VideoCapture(0)

        cv2.namedWindow("Prediction", cv2.WINDOW_NORMAL)

        while True:
            start_time = time.time()

            ret, frame = cap.read()

            if not ret:
                break

            datum = op.Datum()
            datum.cvInputData = frame
            opWrapper.emplaceAndPop(op.VectorDatum([datum]))

            keypoints = datum.poseKeypoints

            if keypoints is not None and keypoints.shape:
                required_indices = [12, 9, 13, 10, 14, 11, 19, 20, 22, 23]
                required_keypoints = keypoints[0, required_indices, :].reshape(1, -1)

                prediction = model.predict(required_keypoints)

                if prediction[0] == 0:
                    result = "INVALID SQUAT"
                else:
                    result = "VALID SQUAT"

                cv2.putText(datum.cvOutputData, result, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

                prediction_frame = np.zeros((200, 400, 3), dtype=np.uint8)
                cv2.putText(prediction_frame, result, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Prediction", prediction_frame)

            end_time = time.time()
            fps = 1 / (end_time - start_time)

            output_frame = datum.cvOutputData

            cv2.putText(output_frame, f"FPS: {fps:.2f}", (output_frame.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if not args[0].no_display:
                cv2.imshow("OpenPose - Squat Analysis", output_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release resources
        cap.release()
    cv2.destroyAllWindows()
except Exception as e:
    print(e)
    sys.exit(-1)