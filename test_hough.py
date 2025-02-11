import cv2
import numpy as np
import os

def detect_pizza(image_path):
    # Read the image
    image = cv2.imread(image_path)
    
    # resize the image if it is too big
    if image.shape[0] > 600 or image.shape[1] > 600:
        rapport = image.shape[0] / image.shape[1]
        image = cv2.resize(image, (600, int(600 * rapport)))
        
    
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Median blur to reduce noise
    blurred = cv2.medianBlur(gray, 5)
    output = blurred.copy()
    
    print("start HoughCircles")
    # Apply Hough Circle Transform
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.5,
        minDist=30,
        param1=50,
        param2=50,
        minRadius= min(image.shape[0], image.shape[1]) // 4,
        maxRadius= min(image.shape[0], image.shape[1]) // 2
    )
    print("HoughCircles terminated")
    
    # Draw detected circles
    if circles is not None:
        circles = np.uint16(np.around(circles))
    else : 
        print("No circle detected")
        return output
        
    # Draw the biggest circles which can be a pizza (the whole circle is on the image)
    circles = sorted(circles[0], key=lambda x: x[2], reverse=True)
    while(circles):
        (x, y, r) = circles.pop(0)
        
        if x < r or y <r or x + r >= image.shape[1] or y + r >= image.shape[0]:
            continue
        cv2.circle(output, (x, y), r, (0, 255, 0), 4)
        break
        
    print(f"Circle detected in {image_path.split('\\')[-1]}")
    return output

# Test the function
if __name__ == "__main__":
    # Replace with your image path
    
    # image_path = "pizzamargherita-20250211T073136Z-001/pizzamargherita/m15.jpg"
    # result = detect_pizza(image_path)
    
    # # Display results
    # cv2.imshow("Detected Pizza", result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    
     # Replace with your image path
    
    dirs = os.listdir("dataset/dataset")
    print(dirs)
    for dir in dirs:
        # Create output directory for the current 'dir'
        output_dir = f"results/{dir}"
        os.makedirs(output_dir, exist_ok=True)
        files = os.listdir(f"dataset/dataset/{dir}")
        print(files)
        for file in files:
            image_path = os.path.join(f"dataset/dataset/{dir}", file)
            result = detect_pizza(image_path)
            cv2.imwrite(os.path.join(output_dir, file), result) 
            # cv2.imshow("Detected Pizza", detect_pizza(image_path))
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

