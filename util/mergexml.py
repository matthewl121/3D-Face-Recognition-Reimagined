import os
import xml.etree.ElementTree as ET

# Directory containing all individual XML files
annotations_dir = "landmarkXML/"
combined_xml_path = "output/combined_annotations.xml"

# Create a root element for the combined XML
root = ET.Element("dataset")

# Add structure for compatibility with Dlib
images = ET.SubElement(root, "images")

# Iterate through all XML files in the annotations folder
for xml_file in os.listdir(annotations_dir):
    if xml_file.endswith(".xml"):
        # Parse each XML file
        xml_path = os.path.join(annotations_dir, xml_file)
        tree = ET.parse(xml_path)
        image_element = tree.getroot().find("image")  # Locate the 'image' element in each XML
        
        # Append each 'image' element to the combined XML's root
        images.append(image_element)

# Write the combined XML to file
tree = ET.ElementTree(root)
tree.write(combined_xml_path)

print(f"Combined XML file created at: {combined_xml_path}")
