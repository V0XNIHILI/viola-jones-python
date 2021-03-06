from typing import List, Tuple

from itertools import combinations

from math import sqrt

import json

import numpy as np

from PIL import Image, ImageDraw


def show_image(img: np.ndarray):
    img = Image.fromarray(img.astype(np.uint8))
    img.show()


def rgb2gray(img: np.unsignedinteger) -> np.unsignedinteger:
    """
    Convert an RGB image to grayscale.

    Algorithm source: https://en.wikipedia.org/wiki/Grayscale#Converting_color_to_grayscale
    """

    # If the image is already grayscale, return it as is
    if len(img.shape) == 2:
        return img

    return np.dot(img[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)


def draw_boxes_around_detections(
    img: Image, detections: List[Tuple[Tuple[int, int], int]]
):
    # Convert image to RGB if it is grayscale so that the boxes drawn have a color
    if img.mode == "L":
        img = img.convert("RGB")

    draw = ImageDraw.Draw(img)

    for detection in detections:
        # Swap x and y because PIL wants the first entry to be x while we have that as y
        left_top = (detection[0][1], detection[0][0])

        right_bottom = (
            left_top[0] + detection[1],
            left_top[1] + detection[1],
        )

        draw.rectangle((left_top, right_bottom), outline="blue")

    return img


def load_cascade(path: str) -> dict:
    with open(path) as json_file:
        return json.load(json_file)["cascade"]


def scale_nearest_neighbor(img: np.unsignedinteger, scale: float) -> np.unsignedinteger:
    """
    Scale an image by nearest neighbor interpolation.

    A scale larger than 1 means that the image is scaled down (ie. smaller).
    """

    new_img = np.zeros(
        (int(img.shape[0] / scale), int(img.shape[1] / scale)), dtype=np.uint8
    )

    for i in range(new_img.shape[0]):
        for j in range(new_img.shape[1]):
            # Make sure that the nearest neighbor pixel in the original image by using min()
            new_img[i, j] = img[
                min(img.shape[0], int(i * scale)), min(img.shape[1], int(j * scale))
            ]

    return new_img


def integral_image(img: np.unsignedinteger, square=False) -> np.unsignedinteger:
    """
    In an integral image each pixel is the sum of all pixels in the original image
    that are 'left and above' the pixel.

    Original    Integral
    +--------   +----------
    | 1 2 3 .   | 1  3  6 .
    | 4 5 6 .   | 5 12 21 .
    | . . . .   | . . . . .

    Algorithm source: https://github.com/scikit-image/scikit-image/blob/main/skimage/transform/integral.py
    Table source: https://github.com/Simon-Hohberg/Viola-Jones/blob/master/violajones/IntegralImage.py
    """

    img = img.astype(np.int64)

    if square == True:
        img = img ** 2

    return np.cumsum(np.cumsum(img, axis=0), axis=1)


def calc_rectangle_feature(
    integral_img: np.unsignedinteger,
    left_top: Tuple[int, int],
    right_bottom: Tuple[int, int],
    weight: int = 1,
) -> int:
    feature = integral_img[right_bottom[0], right_bottom[1]]

    # If-else statements to make sure the calculations do not go out of bounds
    if left_top[0] != 0:
        if left_top[1] != 0:
            feature += integral_img[left_top[0] - 1, left_top[1] - 1]

        feature -= integral_img[left_top[0] - 1, right_bottom[1]]

    if left_top[1] != 0:
        feature -= integral_img[right_bottom[0], left_top[1] - 1]

    return weight * feature


def detect_objects(
    gray_img: np.unsignedinteger,
    cascade: dict,
    detector_size: int = 24,
    scale_factor: float = 1.25,
    delta: float = 1.5,
) -> List[Tuple[Tuple[int, int], int]]:
    step_size = int(delta * scale_factor)

    # Will contain all detected faces.
    detections = []

    current_scale = 1.0

    # Loop over the scales in the image pyramid. Please note that in the original implementation as
    # proposed by Viola and Jones, the detector is scaled instead of the image.
    while (
        gray_img.shape[0] / current_scale >= detector_size
        and gray_img.shape[1] / current_scale >= detector_size
    ):
        current_image = scale_nearest_neighbor(gray_img, current_scale)
        current_integral_image = integral_image(current_image)
        current_squared_integral_image = integral_image(current_image, True)

        # Loop over the image sub windows (i = height, j = width)
        for i in range(0, current_image.shape[0] - detector_size + 1, step_size):
            for j in range(0, current_image.shape[1] - detector_size + 1, step_size):
                # Mean and squared sum calculations are done this way as this approach can also be
                # used in hardware.

                regular_sum = calc_rectangle_feature(
                    current_integral_image,
                    (i, j),
                    (i + detector_size - 2, j + detector_size - 2),
                )

                squared_sum = calc_rectangle_feature(
                    current_squared_integral_image,
                    (i, j),
                    (i + detector_size - 2, j + detector_size - 2),
                )

                variance = squared_sum * (detector_size ** 2) - regular_sum ** 2

                # std = sqrt(true variance) * (detector_size^2); need multiplication with detector
                # size since it will be post-applied
                std = int(sqrt(variance))

                window_left_top = np.array([i, j])

                detected_face = True

                # Loop over the detection cascade
                for layer in cascade:
                    layer_sum = 0

                    # Loop over the features in the current layer of the cascade
                    for feature in layer["features"]:
                        feature_sum = 0

                        for rectangles in feature["rectangles"]:
                            feature_sum += calc_rectangle_feature(
                                current_integral_image,
                                window_left_top + rectangles["left_top"],
                                window_left_top + rectangles["right_bottom"],
                                rectangles["weight"],
                            )

                        # Apply variance normalization as explained in the original paper by
                        # multiplying with the standard deviation of the window.
                        if feature_sum >= feature["threshold"] * std:
                            layer_sum += feature["pass_value"]
                        else:
                            layer_sum += feature["fail_value"]
                    if layer_sum < 0.4 * layer["threshold"]:
                        detected_face = False
                        break

                if detected_face == True:
                    detections.append(
                        (
                            (int(i * current_scale), int(j * current_scale)),
                            int(detector_size * current_scale),
                        )
                    )

        current_scale *= scale_factor

    return detections


def overlap_ratio(
    left_top_1: Tuple[int, int], size_1: int, left_top_2: Tuple[int, int], size_2: int
) -> float:
    """
    Calculate the overlap ratio between two rectangles. The overlap ratio is equal to the
    interected area divided by the union area.

    All coordinates are in the form of (y, x) with the left top corner being (0, 0). All input
    values should consist of integers.

    Algorithm source: https://stackoverflow.com/questions/9324339/how-much-do-two-rectangles-overlap/9325084
    """

    intersection_area = max(
        0,
        min(left_top_1[1] + size_1, left_top_2[1] + size_2)
        - max(left_top_1[1], left_top_2[1]),
    ) * max(
        0,
        min(left_top_1[0] + size_1, left_top_2[0] + size_2)
        - max(left_top_1[0], left_top_2[0]),
    )

    if intersection_area >= size_1 ** 2 or intersection_area >= size_2 ** 2:
        return 1

    union_area = size_1 ** 2 + size_2 ** 2 - intersection_area

    return intersection_area / union_area


def average_detections(
    detections: List[Tuple[Tuple[int, int], int]]
) -> Tuple[Tuple[int, int], int]:
    """
    Calculate the average detection of a list of detections.

    The average detection is calculated by taking the average of the left top corner of the
    detections and the average size of the detections.
    """

    left_top = tuple(
        np.array(
            [
                np.mean([detection[0][0] for detection in detections]),
                np.mean([detection[0][1] for detection in detections]),
            ]
        ).astype(int)
    )

    size = int(np.mean([detection[1] for detection in detections]))

    return left_top, size


def group_objects(
    detections: List[Tuple[Tuple[int, int], int]],
    overlap_threshold: float = 0.4,
    min_neighbors: int = 3,
) -> List[Tuple[Tuple[int, int], int]]:
    """
    Group detections that are close to each other. The overlap threshold is used to determine if two
    detections are close enough to each other to be grouped together as one object.

    An overlap treshold of 1.0 means that two detections have to be exactly the same to be grouped,
    a threshold of 0.5 means that the ratio of intersection area to union area has to be at least
    0.5. 0.0 means that they do not have to overlap in order to be grouped together.
    """

    already_combined_detections = []
    final_detections = []

    for i in range(len(detections)):
        overlapping_rectangles = []

        for j in range(len(detections)):
            if i != j and i not in already_combined_detections:
                if (
                    overlap_ratio(
                        detections[i][0],
                        detections[i][1],
                        detections[j][0],
                        detections[j][1],
                    )
                    > overlap_threshold
                ):
                    overlapping_rectangles.append(j)

        if len(overlapping_rectangles) >= min_neighbors:
            already_combined_detections.extend(overlapping_rectangles)
            already_combined_detections.append(i)

            detections_to_average = [detections[k] for k in overlapping_rectangles]
            detections_to_average.append(detections[i])

            final_detections.append(average_detections(detections_to_average))

    return final_detections


if __name__ == "__main__":
    image_path = "images/man.jpeg"
    cascade_path = "cascades/haarcascade_frontalface_default.json"

    cascade = load_cascade(cascade_path)

    original_image = Image.open(image_path)
    image_array = np.array(original_image)

    # Factor to scale with in every scale of the image pyramid
    scale_factor = 1.2

    # Size of the detector in pixels.
    detector_size = 24

    gray_image = rgb2gray(image_array)

    detections = detect_objects(gray_image, cascade, detector_size, scale_factor, 1.5)
    detections = group_objects(detections)

    image_with_detections = draw_boxes_around_detections(original_image, detections)

    # Show the image with the detected faces.
    image_with_detections.show()

    # Save the image with the detected faces.
    image_with_detections.save("-result.".join(image_path.rsplit(".", 1)))
