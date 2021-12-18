# The simplest Python Viola Jones implementation

I was unable to find a simple, working implementation of [Viola Jones](https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf) in Python. Thus, I decided to write my own.

## Float vs. int design choice

This implementation uses `uint8` and `int` for the data processing. The entire pipeline could also be
made to use floating points, but this would make it harder to port this implementation to hardware
(which was another goal of this project).

## Uses cascade classifier

The cascade classifier that this implementation uses is a pre-trained classifier from the OpenCV
library. The original version can be found [here](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml).

### OpenCV Cascade `xml` format

As there is very little documentation on the format of the OpenCV cascade `xml` files, I have
decoded their meaning by myself:

```xml
<_>
  <rects>
    <_>
      6 5 12 16 -1.</_>
    <_>
      6 13 12 8 2.</_></rects></_>
<_>
```

Each row contains five numbers, which mean the following:

- Left top x
- Left top y
- Width of the rectangle
- Height of the rectangle
- Weight of rectangle (multiplied by 4096, will be explained below)

Next, the following tag indicates the threshold of a stage:

```xml
<stageThreshold>-5.0425500869750977e+00</stageThreshold>
```

This value should be multiplied by 256 to find the threshold we used (as we use `uint8` for the
pixel value instead of a `float` between 0 and 1).

Finally, the weights and thresholds per feature:

```xml
<internalNodes>
  0 -1 0 -3.1511999666690826e-02</internalNodes>
<leafValues>
  2.0875380039215088e+00 -2.2172100543975830e+00</leafValues></_>
```

The internal node has four values, from left to right:

- Left child index
- Right child index
- Feature index
- Node threshold

For the leaf values:

- Value to accumulate if feature is smaller than node threshold
- Value to accumulate if feature is larger than node threshold

The weights and feature thresholds are multiplied by 4096 as they are often in the order of
magnitude of 10^-2. So, in order for these values to be useful in the integer implementation, we
multiply all of these by 4096.

## Image sources:

- [`man.jpeg`](./images/man.jpeg): [source](https://img.freepik.com/free-photo/handsome-young-man-with-new-stylish-haircut_176420-19637.jpg?size=626&ext=jpg)
