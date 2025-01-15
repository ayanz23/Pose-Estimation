import cv2
import PoseModule as pm
import math
import numpy as np


def calculate_angle(a, b, c):
    """Calculate angle between three points in degrees"""
    ba = np.array([a[0] - b[0], a[1] - b[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)


def detect_view_angle(lmList, image_width):
    """Determine if the pose is captured from side view or front view"""
    if len(lmList) < 33:
        return "unknown"

    # Get shoulder landmarks
    left_shoulder = lmList[11]
    right_shoulder = lmList[12]

    # Calculate shoulder width in pixels relative to image width
    shoulder_width = abs(right_shoulder[0] - left_shoulder[0])
    threshold = 0.15 * image_width  # Dynamic threshold based on image width

    # If shoulders are close together horizontally, it's likely a side view
    return "side" if shoulder_width < threshold else "front"


def detect_pose(lmList, view_angle):
    """Detect yoga poses based on the view angle"""
    pose_name = "Unknown Pose"
    confidence = 0

    try:
        if view_angle == "front":
            # Extract landmarks for front view poses
            left_shoulder, right_shoulder = lmList[11], lmList[12]
            left_hip, right_hip = lmList[23], lmList[24]
            left_knee, right_knee = lmList[25], lmList[26]
            left_ankle, right_ankle = lmList[27], lmList[28]

            # Symmetry check
            shoulder_slope = abs(left_shoulder[1] - right_shoulder[1])
            hip_slope = abs(left_hip[1] - right_hip[1])

            # Mountain Pose (Tadasana)
            if (
                shoulder_slope < 20
                and hip_slope < 20
                and abs(left_knee[1] - right_knee[1]) < 20
                and abs(left_ankle[1] - right_ankle[1]) < 20
            ):
                pose_name = "Mountain Pose"
                confidence = 0.9

            # Tree Pose (Vrksasana)
            elif (
                shoulder_slope < 20
                and abs(left_hip[1] - right_hip[1]) < 30
                and (
                    right_knee[1] < right_hip[1] - 50
                    or left_knee[1] < left_hip[1] - 50
                )
            ):
                pose_name = "Tree Pose"
                confidence = 0.85

        elif view_angle == "side":
            # Extract landmarks for side view poses
            shoulder, hip, knee, ankle = lmList[12], lmList[24], lmList[26], lmList[28]

            # Downward Dog
            if (
                hip[1] < shoulder[1]
                and knee[1] > hip[1]
                and ankle[1] > knee[1]
                and calculate_angle(shoulder, hip, ankle) < 70
            ):
                pose_name = "Downward Dog"
                confidence = 0.85

            # Cobra Pose
            elif (
                shoulder[1] < hip[1]
                and knee[1] > hip[1]
                and ankle[1] > knee[1]
                and calculate_angle(shoulder, hip, knee) > 150
            ):
                pose_name = "Cobra Pose"
                confidence = 0.8

    except Exception as e:
        print(f"Error detecting pose: {e}")

    return pose_name, confidence


def process_image(image_path):
    """Process image and detect yoga pose"""
    img = cv2.imread(image_path)
    if img is None:
        print("Failed to load image. Check the file path.")
        return None

    # Resize image if too large
    max_dimension = 1200
    height, width = img.shape[:2]
    if max(height, width) > max_dimension:
        scale = max_dimension / max(height, width)
        img = cv2.resize(img, None, fx=scale, fy=scale)

    # Initialize pose detector
    detector = pm.poseDetector()

    # Perform pose detection
    img = detector.findPose(img)
    lmList = detector.findPosition(img, draw=True)

    # Detect view angle
    view_angle = detect_view_angle(lmList, width)

    # Detect the yoga pose
    pose, confidence = detect_pose(lmList, view_angle)

    # Display the detected pose, confidence, and view angle
    text = f"{pose} ({confidence:.2f}) - {view_angle} view"
    cv2.putText(img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the image with detected pose
    cv2.imshow("Yoga Pose Detection", img)

    # Save the output image
    output_path = f"output_{pose.lower().replace(' ', '_')}.jpg"
    cv2.imwrite(output_path, img)
    print(f"Pose detected: {pose} (Confidence: {confidence:.2f}) from {view_angle} view. Saved result to {output_path}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = "mountain.png"  # Replace with image path
    process_image(image_path)