import cv2
import numpy as np
import os

# Correct file path
file_path = '../misc/areaSegments2.jpg'

# Check if the file exists
if not os.path.isfile(file_path):
    raise FileNotFoundError(f"File does not exist at path: {file_path}")

# Read the image
image = cv2.imread(file_path)

# Check if the image is loaded successfully
if image is None:
    raise ValueError(f"Unable to load image at path: {file_path}. The file might be corrupted or in an unsupported format.")


# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding to keep white and light gray objects
_, thresh = cv2.threshold(gray, 140, 240, cv2.THRESH_BINARY)

# Create a mask of the bright white objects
white_mask = thresh.copy()

cv2.imshow("Detected White Regions", white_mask)

cv2.waitKey(0)

# Apply the mask to the original image
# Invert the white mask to get black regions
black_mask = cv2.bitwise_not(white_mask)

# Find contours in the white mask
contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest contour (assumed to be the big white box)
largest_contour = max(contours, key=cv2.contourArea)

# Create a mask for the largest contour (big white box)
box_mask = np.zeros_like(white_mask)
cv2.drawContours(box_mask, [largest_contour], 0, 255, -1)

# Combine the black mask and the box mask
result_mask = cv2.bitwise_and(black_mask, box_mask)

# Apply the result mask to the original image
result = cv2.bitwise_and(image, image, mask=result_mask)

# Display the image with detected black regions inside the white box
cv2.imshow("Detected Black Regions", result)

cv2.waitKey(0)

# Convert result back to grayscale for further processing
gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

# Apply edge detection
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Apply stronger dilation to connect nearby edges
kernel = np.ones((5,5), np.uint8)
dilated_edges = cv2.dilate(edges, kernel, iterations=2)

# Apply Hough Line Transform with relaxed parameters
lines = cv2.HoughLinesP(dilated_edges, 1, np.pi/180, threshold=30, minLineLength=30, maxLineGap=20)

# Create a blank mask to draw the lines
line_mask = np.zeros_like(gray)

# Draw the detected lines on the mask with thicker lines
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_mask, (x1, y1), (x2, y2), 255, 4)  # Increased thickness to 4

# Apply stronger dilation to further connect the lines
connected_lines = cv2.dilate(line_mask, kernel, iterations=3)

# Apply closing operation to fill small gaps
connected_lines = cv2.morphologyEx(connected_lines, cv2.MORPH_CLOSE, kernel)

# Display the strongly connected lines
# Apply morphological operations to remove small "nugget" shapes
kernel = np.ones((5,5), np.uint8)
cleaned_lines = cv2.morphologyEx(connected_lines, cv2.MORPH_OPEN, kernel)

# Apply skeletonization to thin the lines
def skeletonize(img):
    skel = np.zeros(img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    while True:
        open = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(img, open)
        eroded = cv2.erode(img, element)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return skel

thinned_lines = skeletonize(cleaned_lines)

# Double the thickness of the white lines
# Make lines even thicker
thicker_lines = cv2.dilate(thinned_lines, np.ones((16,16), np.uint8), iterations=2)
# Invert the image
inverted_lines = cv2.bitwise_not(thicker_lines)
# Create a copy of the inverted lines image for flood filling
flood_fill_image = inverted_lines.copy()

# Convert to 3-channel image for colored flood fill
flood_fill_image = cv2.cvtColor(flood_fill_image, cv2.COLOR_GRAY2BGR)

# Get image dimensions
h, w = inverted_lines.shape[:2]

# Create a mask for flood fill, slightly larger than the image
mask = np.zeros((h+2, w+2), np.uint8)

# Flood fill from multiple seed points
for y in range(0, h, 20):  # Adjust step size as needed
    for x in range(0, w, 20):  # Adjust step size as needed
        if inverted_lines[y, x] == 255:  # If the pixel is white (background)
            # Generate a random color
            color = tuple(np.random.randint(0, 255, 3).tolist())
            # Perform flood fill
            cv2.floodFill(flood_fill_image, mask, (x, y), color)

cv2.imshow("Flood Filled Sections", flood_fill_image)

cv2.waitKey(0)

# Convert result back to grayscale for further processing
gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

# Apply edge detection
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Apply Hough Line Transform
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)

# Create a blank mask to draw the lines
line_mask = np.zeros_like(gray)

# Draw the detected lines on the mask
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)

# Display the line mask
cv2.imshow("Detected Lines", line_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
