import argparse
import cv2
import time
import itertools
from imutils import paths

# Parser des arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--models", required=True,
    help="chemin vers le dossier contenant les modèles .t7")
ap.add_argument("-v", "--video",
    help="chemin vers la vidéo en entrée (facultatif, sinon webcam)")
args = vars(ap.parse_args())

# Lister les modèles
modelPaths = sorted(list(paths.list_files(args["models"], validExts=(".t7",))))
modelIter = itertools.cycle(enumerate(modelPaths))  # Boucle infinie sur les modèles
(modelID, modelPath) = next(modelIter)

# Charger le premier modèle
print(f"[INFO] Chargement du modèle {modelID + 1}: {modelPath}")
net = cv2.dnn.readNetFromTorch(modelPath)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)

# Ouvrir la vidéo ou webcam
if args.get("video", None) is None:
    print("[INFO] Streaming depuis la webcam...")
    vs = cv2.VideoCapture(0)
else:
    print(f"[INFO] Ouverture de la vidéo : {args['video']}...")
    vs = cv2.VideoCapture(args["video"])

if not vs.isOpened():
    print("[ERROR] Impossible d'ouvrir la source vidéo")
    exit()

frame_count = 0
while True:
    ret, frame = vs.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 2 != 0:
        continue

    # Redimensionner
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

    print(f"[INFO] Frame stylisée en {end - start:.4f} secondes")

    key = cv2.waitKey(1) & 0xFF

    # q pour quitter
    if key == ord('q'):
        break

    # n pour changer de style
    elif key == ord('n'):
        (modelID, modelPath) = next(modelIter)
        print(f"[INFO] Changement vers le modèle {modelID + 1}: {modelPath}")
        net = cv2.dnn.readNetFromTorch(modelPath)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)

vs.release()
cv2.destroyAllWindows()
