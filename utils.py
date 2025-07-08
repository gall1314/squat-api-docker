import numpy as np

def calculate_angle(a, b, c):
    a, b, c = map(np.array, [a, b, c])
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

def calculate_body_angle(shoulder, hip):
    vector = np.array(shoulder) - np.array(hip)
    vertical = np.array([0, -1])
    norm = np.linalg.norm(vector)
    if norm == 0:
        return 0
    cos_angle = np.dot(vector, vertical) / norm
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
