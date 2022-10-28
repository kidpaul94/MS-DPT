from PIL import Image # (pip install Pillow)
import numpy as np
from skimage import measure                        # (pip install scikit-image)
from shapely.geometry import Polygon, MultiPolygon # (pip install Shapely)
import matplotlib.pyplot as plt
import json
from base64 import b64encode 
from io import BytesIO

def create_sub_masks(mask_image):
    width, height = mask_image.size

    # Initialize a dictionary of sub-masks indexed by RGB colors
    sub_masks = {}
    for x in range(width):
        for y in range(height):
            # Get the RGB values of the pixel
            pixel = mask_image.getpixel((x,y))

            # If the pixel is not black...
            if pixel != 0:
                if (255 - pixel) < 5:
                    sub_mask = sub_masks.get(str(255))
                    # Check to see if we've created a sub-mask...
                    if sub_mask is None:
                        # Create a sub-mask (one bit per pixel) and add to the dictionary
                        # Note: we add 1 pixel of padding in each direction because the contours module doesn't handle cases 
                        # where pixels bleed to the edge of the image
                        sub_masks[str(255)] = Image.new('1', (width+2, height+2))
                    # Set the pixel value to 1 (default is 0), accounting for padding
                    sub_masks[str(255)].putpixel((x+1, y+1), 1)
                elif abs(128 - pixel) < 5:
                    sub_mask = sub_masks.get(str(128))     
                    if sub_mask is None:
                        sub_masks[str(128)] = Image.new('1', (width+2, height+2))
                    sub_masks[str(128)].putpixel((x+1, y+1), 1)

    return sub_masks

def create_sub_mask_annotation(sub_mask):
    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)
    contours = measure.find_contours(sub_mask, 0.5, positive_orientation='low')
    
    switch = 0
    segmentations = []
    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        # Make a polygon and simplify it
        poly = Polygon(contour)
        poly = poly.simplify(1.0, preserve_topology=False)
        # polygons.append(poly)
        segmentation = np.array(poly.exterior.coords)

        if switch == 0:
            segmentations = segmentation
            switch = 1
        elif len(segmentations) < len(segmentation):
            segmentations = segmentation
        
        #if switch == 0:
        #    segmentations = segmentation
        #    switch = 1
        #else:
        #    segmentations = np.concatenate([segmentations, segmentation])

    return segmentations

# Looping to generate multiple annotations at once
for i in range(1000):
    substring = str(i+1)
    string = '/home/kidpaul/Python/validation/Mask_' + substring + '0002.jpg'
    mask_image = Image.open(string)
    sub_masks = create_sub_masks(mask_image)
    polygons1 = create_sub_mask_annotation(np.array(sub_masks['255']))
    polygons2 = create_sub_mask_annotation(np.array(sub_masks['128']))

    # Debugging/Visualization
    """
    num = len(polygons1)
    x = []
    y = []
    for i in range(num):
        x.append(polygons1[i][0])
        y.append(polygons1[i][1])

    print(x)
    print(y)
    print(polygons1)
    plt.plot(x, y)
    plt.axis([0, 550, 0, 550])
    plt.show()
    """

    # Labelme .json format 
    data = {
        "version": "4.5.7",
        "flags": {}
    }
    data["shapes"] = [] 
    data["shapes"].append({
        "label": "Object1",
        "points": polygons1.tolist(),
        "group_id": None,
        "shape_type": "polygon",
        "flags": {}
    })
    data["shapes"].append({
        "label": "Object1",
        "points": polygons2.tolist(),
        "group_id": None,
        "shape_type": "polygon",
        "flags": {}
    })

    original_image = Image.open('/home/kidpaul/Python/validation/Image_' + substring + '.jpg')
    # Create a buffer to hold the bytes
    buf = BytesIO()
    # Save the image as jpg to the buffer
    original_image.save(buf, 'jpeg')
    # Rewind the buffer's file pointer
    buf.seek(0)
    # Read the bytes from the buffer
    image_bytes = buf.read()
    # Close the buffer
    buf.close()

    encoded_string = b64encode(image_bytes)

    data["imagePath"] = "Image_" + substring + ".jpg"
    data["imageData"] = encoded_string.decode('utf-8')
    data["imageHeight"] = 550
    data["imageWidth"] = 550

    name = 'Image_' + substring + '.json'
    with open(name, 'w') as outfile:
        json.dump(data, outfile, indent = 2)
