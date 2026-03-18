# This class helps guide the user to properly position text on screen for reading
class GuidanceSystem:
    # Initialize with frame size
    def __init__(self, frame_width=640, frame_height=480):
        self.w = frame_width
        self.h = frame_height
        self._update_bounds()

    # Calculate the center zone where text should be positioned (middle 30% of screen)
    def _update_bounds(self):
        self.center_x_min = self.w * 0.35
        self.center_x_max = self.w * 0.65
        self.center_y_min = self.h * 0.35
        self.center_y_max = self.h * 0.65

    # Update the screen size when a new camera frame comes in
    def update_frame_dims(self, frame):
        if frame is not None:
            self.h, self.w = frame.shape[:2]
            self._update_bounds()

    # Check where the text box is on screen and tell user how to move camera
    # box is [x1, y1, x2, y2] coordinates of text or None if not found
    def get_guidance(self, box):
        # No text found on screen
        if box is None or len(box) == 0:
            return "No object detected. Please scan the surroundings."

        # Find the center point of the text box
        obj_x_center = (box[0] + box[2]) / 2
        obj_y_center = (box[1] + box[3]) / 2

        # Check if text is not in the center zone and give directions
        if obj_x_center < self.center_x_min:
            return "Move the camera slightly to the left."
        if obj_x_center > self.center_x_max:
            return "Move the camera slightly to the right."
        if obj_y_center < self.center_y_min:
            return "Tilt the camera up."
        if obj_y_center > self.center_y_max:
            return "Tilt the camera down."

        # Check if text is too small - user needs to move camera closer
        box_width = box[2] - box[0]
        if box_width < (self.w * 0.2):
            return "Bring the camera closer to the object."

        # Text is centered and big enough - ready to read
        return "OK"