import json
import boto3
import onnxruntime as ort
import numpy as np
from PIL import Image
from io import BytesIO
from datetime import datetime
import psycopg2
import os


# Initialize S3 client and ONNX session globally
s3_client = boto3.client('s3')
model_path = '/opt/python/models/yolov8n.onnx'
session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

# RDS connection details
RDS_HOST = os.environ['DB_PROXY_ENDPOINT']
RDS_PORT = os.environ.get('DB_PORT', '5432')
RDS_USER = os.environ['DB_USER']
RDS_PASSWORD = os.environ['DB_PASSWORD']
RDS_DB = os.environ['DB_NAME']

# Parameters
CONFIDENCE_THRESHOLD = 0.7
IOU_THRESHOLD = 0.5


def non_max_suppression(boxes, scores, iou_threshold):
    """Apply Non-Maximum Suppression"""
    if len(boxes) == 0:
        return []

    # Convert to numpy arrays
    boxes = np.array(boxes)
    scores = np.array(scores)

    # Get indices sorted by scores (descending)
    indices = np.argsort(scores)[::-1]

    keep = []
    while len(indices) > 0:
        # Keep the detection with highest score
        current = indices[0]
        keep.append(current)

        if len(indices) == 1:
            break

        # Calculate IoU with remaining detections
        current_box = boxes[current]
        remaining_boxes = boxes[indices[1:]]

        # Calculate intersection
        x1 = np.maximum(current_box[0], remaining_boxes[:, 0])
        y1 = np.maximum(current_box[1], remaining_boxes[:, 1])
        x2 = np.minimum(current_box[2], remaining_boxes[:, 2])
        y2 = np.minimum(current_box[3], remaining_boxes[:, 3])

        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

        # Calculate union
        current_area = (current_box[2] - current_box[0]) * \
            (current_box[3] - current_box[1])
        remaining_areas = (remaining_boxes[:, 2] - remaining_boxes[:, 0]) * (
            remaining_boxes[:, 3] - remaining_boxes[:, 1])
        union = current_area + remaining_areas - intersection

        # Calculate IoU
        iou = intersection / union

        # Keep detections with IoU below threshold
        indices = indices[1:][iou < iou_threshold]

    return keep

# we are creating a 4d tensor here which the yolo model expects as input
def preprocess_image(image):
    image = image.convert('RGB')
    image = image.resize((640, 640))
    image_np = np.array(image).astype(np.float32) / 255.0
    image_np = image_np.transpose(2, 0, 1)  # CHW format
    image_np = np.expand_dims(image_np, axis=0)
    return image_np


def write_detection_to_db(image_key, timestamp, location, speed, classification, pedestrians_detected):
    try:

        # Connect to RDS
        conn = psycopg2.connect(
            host=RDS_HOST,
            port=RDS_PORT,
            user=RDS_USER,
            password=RDS_PASSWORD,
            database=RDS_DB,
            connect_timeout=5
        )

        cursor = conn.cursor()

        # Insert into DB
        cursor.execute(
            """
            INSERT INTO vehicle_safety_data 
            (image_key, timestamp, location, speed, classification, pedestrians_detected)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id;
            """,
            (image_key, timestamp, location, speed,
             classification, pedestrians_detected)
        )

        new_id = cursor.fetchone()[0]

        conn.commit()
        cursor.close()
        conn.close()

        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Write successful',
                'record_id': new_id,
                'inserted_data': {
                    'image_key': image_key,
                    'timestamp': timestamp.isoformat(),
                    'location': location,
                    'speed': speed,
                    'classification': classification,
                    'pedestrians_detected': pedestrians_detected
                }
            })
        }

    except Exception as e:
        print(f"Write failed: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps(f'Error writing to RDS: {str(e)}')
        }


def lambda_handler(event, context):
    try:
        print(f"Received event: {json.dumps(event)}")

        for record in event['Records']:
            bucket = record['s3']['bucket']['name']
            key = record['s3']['object']['key']
            print(f"Processing bucket: {bucket}, key: {key}")

            # Download image
            response = s3_client.get_object(Bucket=bucket, Key=key)
            content_type = response.get('ContentType', 'unknown')
            print(f"Content-Type: {content_type}")

            image_data = response['Body'].read()
            print(f"Image data size: {len(image_data)} bytes")

            if not content_type.startswith('image/'):
                raise ValueError(
                    f"File {key} is not an image, Content-Type: {content_type}")

            image = Image.open(BytesIO(image_data))

            # Preprocess
            input_data = preprocess_image(image)

            # Inference
            input_name = session.get_inputs()[0].name
            outputs = session.run(None, {input_name: input_data})

            # Process YOLOv8 output
            predictions = outputs[0]
            print(f"Model output shape: {predictions.shape}")

            detections = predictions[0].T

            # Collect all detections above threshold
            pedestrian_boxes = []
            pedestrian_scores = []
            all_detections = []

            for det in detections:
                x, y, w, h = det[:4]
                class_probs = det[4:]
                class_id = int(np.argmax(class_probs))
                confidence = float(class_probs[class_id])

                if confidence > CONFIDENCE_THRESHOLD:
                    # Convert to corner format for NMS
                    x1, y1 = x - w/2, y - h/2
                    x2, y2 = x + w/2, y + h/2

                    all_detections.append({
                        'class_id': class_id,
                        'confidence': confidence,
                        'box': [x1, y1, x2, y2]
                    })

                    if class_id == 0:  # Person class
                        pedestrian_boxes.append([x1, y1, x2, y2])
                        pedestrian_scores.append(confidence)

            # Apply NMS to pedestrian detections
            pedestrian_detected = False
            pedestrian_count = 0

            if pedestrian_boxes:
                keep_indices = non_max_suppression(
                    pedestrian_boxes, pedestrian_scores, IOU_THRESHOLD)
                pedestrian_count = len(keep_indices)
                pedestrian_detected = pedestrian_count > 0

                print(f"Pedestrians after NMS: {pedestrian_count}")
                for i in keep_indices:
                    print(
                        f"Pedestrian detected with confidence {pedestrian_scores[i]:.3f}")

            print(
                f"Raw detections: {len(all_detections)}, Pedestrians after NMS: {pedestrian_count}")

            location = 'TestLocation'
            speed = 45.0
            result = write_detection_to_db(key, datetime.utcnow(
            ), location, speed, 'critical' if pedestrian_count > 0 else 'safe', pedestrian_count)
            print(result)
            return result

    except Exception as e:
        error_result = {
            'statusCode': 500,
            'body': json.dumps(f'Error processing image: {str(e)}')
        }
        print(error_result)
        return error_result
