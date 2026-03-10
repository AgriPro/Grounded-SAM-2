import numpy as np

def sample_points_from_masks(masks, num_points):
    """
    sample points from masks and return its absolute coordinates

    Args:
        masks: np.array with shape (n, h, w)
        num_points: int

    Returns:
        points: np.array with shape (n, points, 2)
    """
    n, h, w = masks.shape
    points = []

    for i in range(n):
        # find the valid mask points
        indices = np.argwhere(masks[i] == 1)  
        # the output format of np.argwhere is (y, x) and the shape is (num_points, 2)
        # we should convert it to (x, y)
        indices = indices[:, ::-1]  # (num_points, [y x]) to (num_points, [x y])
        
        # import pdb; pdb.set_trace()
        if len(indices) == 0:
            # if there are no valid points, append an empty array
            points.append(np.array([]))
            continue
        
        # resampling if there's not enough points
        if len(indices) < num_points:
            sampled_indices = np.random.choice(len(indices), num_points, replace=True)
        else:
            sampled_indices = np.random.choice(len(indices), num_points, replace=False)
        
        sampled_points = indices[sampled_indices]
        points.append(sampled_points)

    # convert to np.array
    points = np.array(points, dtype=np.float32)
    return points


def masks_to_boxes(masks, obj_ids, min_area=20):
    """
    Convert masks → boxes aligned to obj_ids order.

    Returns:
        boxes: np.ndarray (N, 4)
        valid_obj_ids: list
    """
    boxes = []
    valid_obj_ids = []
    for i, obj_id in enumerate(obj_ids):
        mask = masks[i]
        ys, xs = np.where(mask > 0)
        if len(xs) < min_area:
            raise RuntimeError("Mask missing at chunk boundary")
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        boxes.append([x1, y1, x2, y2])
        valid_obj_ids.append(obj_id)
    if len(boxes) == 0:
        return None, None
    return np.array(boxes), valid_obj_ids