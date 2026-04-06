# Detected person in camera frame
std_msgs/Header header
float64 bearing_rad          # horizontal angle to person from robot center
float64 distance_m           # estimated distance in meters (0 if unknown)
float64 confidence           # 0.0 to 1.0
bool detected                # true if a person is currently detected
