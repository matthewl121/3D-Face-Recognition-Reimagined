import xml.etree.ElementTree as ET
import cv2
import os

image_folder = r'C:\Users\mli00\Desktop\3D_Pictures'  # Folder with images
annotation_folder = 'landmarkXML'  # Folder with XML files

# Loop through XML files and parse each annotation
for xml_file in os.listdir(annotation_folder):
    if xml_file.endswith('.xml'):
        tree = ET.parse(os.path.join(annotation_folder, xml_file))
        root = tree.getroot()

        # Read the image
        image_name = root.find('filename').text
        image_path = os.path.join(image_folder, image_name)
        image = cv2.imread(image_path)
        
        # Get the bounding box coordinates
        for obj in root.findall('object'):
            label = obj.find('name').text
            if label == 'face':  # You can customize this
                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)

                # Crop and save the positive samples
                cropped_face = image[ymin:ymax, xmin:xmax]
                cv2.imwrite(f'positive_samples/{image_name}_{label}.jpg', cropped_face)
                
                # You can also prepare negative samples by including random crops from other images
