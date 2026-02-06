#!/usr/bin/env python3
import sys
import os
import cv2
import numpy as np

def main():
    if len(sys.argv) != 2:
        print("Usage: python imageHeader.py <image.png>")
        sys.exit(1)

    img_path = sys.argv[1]
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    header_name = base_name + ".h"

    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")

    height, width = img.shape[:2]

    if img.ndim == 2:
        channels = 1
        cv_type = "CV_8UC1"
    else:
        channels = img.shape[2]
        cv_type = f"CV_8UC{channels}"

    data = img.flatten()

    with open(header_name, "w") as f:
        f.write(f"""#pragma once
#include <cstdint>
#include <opencv2/core.hpp>

/*
  Auto-generated from: {img_path}

  Usage example:

    #include "{header_name}"

    cv::Mat image(
        IMAGE_{base_name.upper()}_HEIGHT,
        IMAGE_{base_name.upper()}_WIDTH,
        IMAGE_{base_name.upper()}_TYPE,
        (void*)IMAGE_{base_name.upper()}_DATA
    );

  Note:
  - The data is stored in row-major order.
  - Color images are in OpenCV default format (BGR or BGRA).
*/

static constexpr int IMAGE_{base_name.upper()}_WIDTH  = {width};
static constexpr int IMAGE_{base_name.upper()}_HEIGHT = {height};
static constexpr int IMAGE_{base_name.upper()}_CHANNELS = {channels};
static constexpr int IMAGE_{base_name.upper()}_TYPE = {cv_type};

static const uint8_t IMAGE_{base_name.upper()}_DATA[] = {{
""")

        for i, byte in enumerate(data):
            if i % 12 == 0:
                f.write("    ")
            f.write(f"0x{byte:02X}, ")
            if i % 12 == 11:
                f.write("\n")

        f.write("""
};
""")

    print(f"Generated header: {header_name}")

if __name__ == "__main__":
    main()
