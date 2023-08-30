from typing import List
import numpy as np


def transformation_matrix_from_t_and_q(t: List[float], q: List[float]) -> np.ndarray:
    """transformation_matrix_from_t_and_q.

    :param t: translation vector [tx, ty, tz]
    :type t: List[float]
    :param q: quaternion [qw, qx, qy, qz]
    :type q: List[float]
    :rtype: np.ndarray
        Transformation matrix resulting from t and q
    """
    # Extract the values from q
    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]

    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)

    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)

    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1

    # Create the trasnformation matrix
    return np.array(
        [
            [r00, r01, r02, t[0]],
            [r10, r11, r12, t[1]],
            [r20, r21, r22, t[2]],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
