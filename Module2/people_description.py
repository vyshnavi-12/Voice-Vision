import cv2
import face_recognition
import numpy as np

class PersonDescriptionModule:
    def get_clothing_color(self, frame, face_location):
        top, right, bottom, left = face_location
        f_height, _, _ = frame.shape
        shirt_top = min(bottom + 20, f_height - 1)
        shirt_bottom = min(bottom + 120, f_height)
        shirt_roi = frame[shirt_top:shirt_bottom, left:right]
        if shirt_roi.size > 0:
            avg_color = np.mean(shirt_roi.reshape(-1, 3), axis=0)
            b, g, r = avg_color
            if r > 180 and g < 120: return "red"
            if b > 180: return "blue"
            if g > 160 and r < 120: return "green"
            if r > 200 and g > 200 and b > 200: return "white"
            if r < 60 and g < 60 and b < 60: return "dark/black"
            return "neutral-colored"
        return "unknown"

    def get_relative_height(self, face_location, frame_height):
        top, _, bottom, _ = face_location
        face_center_y = (top + bottom) / 2
        if face_center_y < frame_height * 0.35: return "taller than baseline"
        if face_center_y > frame_height * 0.6: return "shorter than baseline"
        return "average height"

# --- INDEPENDENT EXECUTION ---
if __name__ == "__main__":
    describer = PersonDescriptionModule()
    cap = cv2.VideoCapture(0)
    print("Person Description Mode Active. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locs = face_recognition.face_locations(rgb)
        for loc in locs:
            color = describer.get_clothing_color(frame, loc)
            height = describer.get_relative_height(loc, frame.shape[0])
            label = f"Shirt: {color} | {height}"
            cv2.rectangle(frame, (loc[3], loc[0]), (loc[1], loc[2]), (255, 0, 0), 2)
            cv2.putText(frame, label, (loc[3], loc[2] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("Person Description Mode", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cap.release()
    cv2.destroyAllWindows()