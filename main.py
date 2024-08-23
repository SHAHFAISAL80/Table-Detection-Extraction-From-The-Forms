
import cv2
import numpy as np
import os

def table_detection(img_path):
    # Read and preprocess the image
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_bin = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img_bin = cv2.bitwise_not(img_bin)

    # Detect vertical lines
    kernel_length_v = np.array(img_gray).shape[1] // 120
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length_v)) 
    im_temp1 = cv2.erode(img_bin, vertical_kernel, iterations=3)
    vertical_lines_img = cv2.dilate(im_temp1, vertical_kernel, iterations=3)

    # Detect horizontal lines
    kernel_length_h = np.array(img_gray).shape[1] // 40
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length_h, 1))
    im_temp2 = cv2.erode(img_bin, horizontal_kernel, iterations=3)
    horizontal_lines_img = cv2.dilate(im_temp2, horizontal_kernel, iterations=3)

    # Combine the vertical and horizontal lines to detect tables
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    table_segment = cv2.addWeighted(vertical_lines_img, 0.5, horizontal_lines_img, 0.5, 0.0)
    table_segment = cv2.erode(cv2.bitwise_not(table_segment), kernel, iterations=2)
    _, table_segment = cv2.threshold(table_segment, 0, 255, cv2.THRESH_OTSU)
   
    # Detect contours
    contours, _ = cv2.findContours(table_segment, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    count = 0

    # Create directories for saving results if they don't exist
    if not os.path.exists(r'F:\152\saba projects\Table-Detection-Extraction-master\results\cropped'):
        os.makedirs('./results/cropped')
    if not os.path.exists(r'F:\152\saba projects\Table-Detection-Extraction-master\results\table_detect'):
        os.makedirs('./results/table_detect')
    if not os.path.exists(r'F:\152\saba projects\Table-Detection-Extraction-master\results\bb'):
        os.makedirs('./results/bb')

    # Process contours and save cropped tables and images with bounding boxes
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if (w > 80 and h > 20) and w > 3 * h:
            count += 1
            cropped = img[y:y + h, x:x + w]
            cv2.imwrite(f"./results/cropped/crop_{count}__{os.path.basename(img_path)}", cropped)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Save the table segment and bounding box images
    cv2.imwrite(f"./results/table_detect/table_detect__{os.path.basename(img_path)}", table_segment)
    cv2.imwrite(f"./results/bb/bb__{os.path.basename(img_path)}", img)

# Process all images in the 'forms' directory
for img_file in os.listdir(r'F:\152\saba projects\Table-Detection-Extraction-master\forms'):
    if img_file.endswith(('.png', '.PNG', '.jpg', '.JPG')):
        table_detection(f'./forms/{img_file}')
