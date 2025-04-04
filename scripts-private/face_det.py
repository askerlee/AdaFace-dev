import insightface
import cv2
import sys

detector = insightface.model_zoo.get_model('models/insightface/models/antelopev2/scrfd_10g_bnkps.onnx')
detector.prepare(ctx_id=0, input_size=(640, 640))

img = cv2.imread(sys.argv[1])
faces = detector.detect(img)
'''
(Pdb) faces
(array([[5.0291425e+02, 3.2051974e+02, 1.4214452e+03, 1.7183633e+03,
        7.6358473e-01]], dtype=float32), array([[[ 722.8099 ,  923.53595],
        [1181.4635 ,  910.7113 ],
        [ 956.15045, 1222.0217 ],
        [ 810.3346 , 1432.7496 ],
        [1134.5233 , 1419.9358 ]]], dtype=float32))
faces[0]: A bounding box for the face: [x, y, width, height, confidence_score]
faces[1]: Five facial landmarks (likely representing eye corners, nose, and mouth corners)
'''

if len(faces[0]) == 0:
    print("No face detected")
else:
    # Draw the bounding box and landmarks on the image
    for i in range(len(faces[0])):
        bbox = faces[0][i]
        x1, y1, x2, y2, score = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), bbox[4]
        print(f"Face {i}: Bounding box: ({x1}, {y1}), ({x2}, {y2}), Score: {score}")
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        landmarks = faces[1][i]
        for landmark in landmarks:
            cv2.circle(img, tuple(map(int, landmark)), 3, (255, 0, 0), -1)

    # Save the image with bounding boxes and landmarks
    output_path = "output.jpg"
    cv2.imwrite(output_path, img)
    print(f"Output saved to {output_path}")
