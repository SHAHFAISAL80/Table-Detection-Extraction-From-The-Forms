# Table-Detection-Extraction-From-The-Forms
Functionality:
Detects all the tables in a form page.
Create bounding boxes around it.
Segment it out and extract the cells of the tables.
Steps:
Grayscale the image
Binary Thresholding
Get all the vertical lines using vertical kernel and cv2.getStructuringElement
Similarly, get all the horizontal lines using horizontal kernel and cv2getStructuringElement
Combine all the horizontal and vertical lines using cv2.addWeighted
Perform some morphological transformation like cv2.erode to get crisp lines & for better results.
Finding the contours and extracting out the rectangles/table cells.
  
