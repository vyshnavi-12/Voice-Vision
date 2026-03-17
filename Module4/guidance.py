class GuidanceSystem:
    def __init__(self, frame_width=640, frame_height=480):
        self.w = frame_width
        self.h = frame_height
        self._update_bounds()

    def _update_bounds(self):
        self.center_x_min = self.w * 0.35
        self.center_x_max = self.w * 0.65
        self.center_y_min = self.h * 0.35
        self.center_y_max = self.h * 0.65

    def update_frame_dims(self, frame):
        if frame is not None:
            self.h, self.w = frame.shape[:2]
            self._update_bounds()

    def get_guidance(self, box):
        """
        box: [x1, y1, x2, y2]  or None
        Returns guidance string or "OK"
        """
        if box is None or len(box) == 0:
            return "No object detected. Please scan the surroundings."

        obj_x_center = (box[0] + box[2]) / 2
        obj_y_center = (box[1] + box[3]) / 2

        if obj_x_center < self.center_x_min:
            return "Move the camera slightly to the left."
        if obj_x_center > self.center_x_max:
            return "Move the camera slightly to the right."
        if obj_y_center < self.center_y_min:
            return "Tilt the camera up."
        if obj_y_center > self.center_y_max:
            return "Tilt the camera down."

        box_width = box[2] - box[0]
        if box_width < (self.w * 0.2):
            return "Bring the camera closer to the object."

        return "OK"