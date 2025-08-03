import argparse
import cv2
import time

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
    help="neural style transfer model")
ap.add_argument("-v", "--video",
    help="path to input video file (optional, if not specified webcam is used)")
args = vars(ap.parse_args())

print("[INFO] loading style transfer model...")
net = cv2.dnn.readNetFromTorch(args["model"])
# Activer le GPU OpenCL 
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)

if args.get("video", None) is None:
    print("[INFO] starting video stream from webcam...")
    vs = cv2.VideoCapture(0)
else:
    print(f"[INFO] opening video file {args['video']}...")
    vs = cv2.VideoCapture(args["video"])

if not vs.isOpened():
    print("[ERROR] Could not open video source")
    exit()

frame_count = 0
while True:
    ret, frame = vs.read()
    if not ret:
        break

    frame_count += 1

    # Pour améliorer la vitesse, on peut traiter une frame sur 2 (skip frames)
    if frame_count % 2 != 0:
        continue

    # Reduire la résolution pour aller plus vite
    frame = cv2.resize(frame, (400, int(frame.shape[0] * 400 / frame.shape[1])))
    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1.0, (w, h),
                                 (103.939, 116.779, 123.680),
                                 swapRB=False, crop=False)
    net.setInput(blob)

    start = time.time()
    output = net.forward()
    end = time.time()

    output = output.reshape((3, output.shape[2], output.shape[3]))
    output[0] += 103.939
    output[1] += 116.779
    output[2] += 123.680
    output /= 255.0
    output = output.transpose(1, 2, 0)
    output = output[:, :, ::-1]

    output = (output * 255).clip(0, 255).astype("uint8")

    cv2.imshow("Input", frame)
    cv2.imshow("Styled Output", output)

    print(f"[INFO] frame processed in {end - start:.4f} seconds")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vs.release()
cv2.destroyAllWindows()
