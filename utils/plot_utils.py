import cv2

def draw_counter(frame, text, origin=(10, 30)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    padding = 6

    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = origin

    # white background rectangle
    cv2.rectangle(
        frame,
        (x - padding, y - th - padding),
        (x + tw + padding, y + padding),
        (255, 255, 255),
        -1
    )
    # black text
    cv2.putText(
        frame,
        text,
        (x, y),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA
    )