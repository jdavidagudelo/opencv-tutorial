import cv2


def draw_line(img, x0, y0, x1, y1, red, green, blue, thickness):
    return cv2.line(img, (x0, y0), (x1, y1), (blue, red, green), thickness)


def draw_rectangle(img, x0, y0, x1, y1, red, green, blue, thickness):
    return cv2.rectangle(img, (x0, y0), (x1, y1), (blue, red, green), thickness)


def draw_circle(img, x0, y0, radius, red, green, blue, thickness):
    return cv2.circle(img, (x0, y0), radius, (blue, red, green), thickness)


def draw_ellipse(img, x0, y0, x1, y1, angle, start_angle, end_angle, red, green, blue, thickness):
    return cv2.ellipse(img, (x0, y0), (x1, y1), angle, start_angle, end_angle, (blue, red, green), thickness)


def draw_polygon(img, points, is_closed, red, green, blue, thickness):
    return cv2.polylines(img, points, is_closed, (blue, red, green), thickness)


def draw_text(img, text, x0, y0, red, green, blue, font_scale, thickness, line_type=cv2.LINE_AA,
              font=cv2.FONT_HERSHEY_SIMPLEX):
    return cv2.putText(img, text, (x0, y0), font, font_scale, (blue, red, green), thickness, line_type)