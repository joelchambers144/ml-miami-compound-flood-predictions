import tensorflow as tf

# Module-level variables
_Y_MIN = None
_Y_MAX = None

def set_y_bounds(y_min_val, y_max_val, buffer=0.2):
    """Set y bounds with an optional extrapolation buffer (default 20%)."""
    global _Y_MIN, _Y_MAX
    _Y_MIN = tf.constant(y_min_val, dtype=tf.float32)
    _Y_MAX = tf.constant(y_max_val * (1 + buffer), dtype=tf.float32)


def get_y_bounds():
    """Return the TensorFlow constants if already set."""
    if _Y_MIN is None or _Y_MAX is None:
        raise ValueError("y_min and y_max not set. Call set_y_bounds() first.")
    return _Y_MIN, _Y_MAX
