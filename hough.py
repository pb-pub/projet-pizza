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


"""Mask the detected pizza"""
def mask_pizza(image_path = None, image = None):
    # Read the image
    if image_path is not None:
        image = cv2.imread(image_path)
    
    initial_shape = image.shape
    resized = False
    
    # resize the image if it is too big
    if image.shape[0] > 600 or image.shape[1] > 600:
        resized = True
        rapport = image.shape[0] / image.shape[1]
        image = cv2.resize(image, (600, int(600 * rapport)))
        
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Median blur to reduce noise
    blurred = cv2.medianBlur(gray, 5)
    
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
    
    # Draw detected circles
    if circles is not None:
        circles = np.uint16(np.around(circles))
    else : 
        print("No circle detected")
        return image
        
    # Draw the biggest circles which can be a pizza (the whole circle is on the image)
    circles = sorted(circles[0], key=lambda x: x[2], reverse=True)
    while(circles):
        (x, y, r) = circles.pop(0)
        
        if x < r or y <r or x + r >= image.shape[1] or y + r >= image.shape[0]:
            continue
        # Create a mask
        mask = np.zeros_like(image)
        cv2.circle(mask, (x, y), int(r*0.95), (255, 255, 255), -1)
        
        if resized:
            # Resize the mask to its initial shape
            mask = cv2.resize(mask, (initial_shape[1], initial_shape[0]))
        
        image = cv2.imread(image_path)
        
        # Apply the mask
        masked_image = cv2.bitwise_and(image, mask)
        
        # crop the image around the circle by finding the first non zero pixel
        # Find the bounding box of non-zero pixels
        coords = cv2.findNonZero(cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY))
        x, y, w, h = cv2.boundingRect(coords)

        # Crop the image to the bounding box
        masked_image = masked_image[y:y+h, x:x+w]
        
        break
        
    print(f"Circle detected in {image_path.split('\\')[-1]}")
     
        
    return masked_image
        

def save_masked_image(image):
    """mask and image and save it to 'masked/pizza_type/pizza_name.jpg' directory"""
    image = mask_pizza(image)
    # print(f"masked/{image_path.split('/')[2]}")
    cv2.imwrite(f"masked/{image_path.split('/')[2]}", image)
    

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
    dirs = os.listdir("./dataset")
    
    
    print(dirs)
    for dir in dirs:
        # Create output directory for the current 'dir'
        output_dir = f"results/{dir}"
        os.makedirs(output_dir, exist_ok=True)
        files = os.listdir(f"./dataset/{dir}")
        print(files)
        for file in files:
            image_path = os.path.join(f"./dataset/{dir}", file)
            
            # Save masked image
            save_masked_image(image_path)
            
            # Save pizza circled
            # result = detect_pizza(image_path)
            # cv2.imwrite(os.path.join(output_dir, file), result) *
            
            # Display results
            # cv2.imshow("Detected Pizza", detect_pizza(image_path))
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

