 from flask import Flask , request , jsonify
 from flask_cors import CORS
 from ultralytics import YOLO
 from PIL import Image
 import io , base64 , cv2 , numpy as np

 app = Flask ( __name__ )
 CORS ( app )
 model = YOLO (" yolov8n .pt") # ~6 MB , downloads on first run

 @app . route ("/ detect ", methods =[" POST "])
 def detect () :
 file = request . files [" image "]
 pil_img = Image . open ( io . BytesIO ( file . read () )). convert (" RGB ")
 result = model ( pil_img ) [0] # single - pass YOLOv8 inference
 cv_img , detections = cv2 . cvtColor ( np . array ( pil_img ) ,
 cv2 . COLOR_RGB2BGR ) , []
 for box in result . boxes :
 x1 ,y1 , x2 , y2 = map(int , box . xyxy [0])
 conf = float ( box . conf [0])
 label = model . names [ int ( box . cls [0]) ]
 if conf > 0.4:
 detections . append ({" label ": label ,
 " confidence ": round ( conf *100 ,1) ,
 "box": [ x1 ,y1 ,x2 , y2 ]})
 cv2 . rectangle ( cv_img ,( x1 , y1 ) ,( x2 , y2 ) ,(0 ,255 ,100) ,2)
 cv2 . putText ( cv_img ,f"{ label } { round ( conf *100) }%",
 ( x1 ,y1 -8) , cv2 . FONT_HERSHEY_SIMPLEX ,0.6 ,
 (0 ,255 ,100) ,2)
 _ , buf = cv2 . imencode (". jpg ", cv_img )
 encoded = base64 . b64encode ( buf ). decode ()
 return jsonify ({" detections ": detections ,
 " image ": f" data : image / jpeg ; base64 ,{ encoded }"})

 if __name__ == " __main__ ":
 app . run ( debug = True , port =5000)