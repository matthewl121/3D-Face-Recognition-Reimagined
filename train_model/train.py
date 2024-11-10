import dlib
import os
import random

# Paths to the XML annotation file and where to save the trained .dat model
xml_annotations = "path/to/your_face_annotations.xml"  # XML file generated from LabelImg
model_output_path = "output/your_face_landmarks_model.dat"  # Output file path for the trained model

# Check if the annotation XML exists
if not os.path.exists(xml_annotations):
    raise FileNotFoundError(f"XML file not found at {xml_annotations}")

# Set training options for Dlib’s shape predictor
options = dlib.shape_predictor_training_options()
options.tree_depth = 4                 # Depth of regression trees; higher values improve accuracy, lower improves speed
options.nu = 0.1                       # Regularization parameter
options.cascade_depth = 15             # Number of cascades (depth affects detection accuracy)
options.feature_pool_size = 400        # Size of the feature pool for each cascade
options.num_test_splits = 50           # Number of test splits
options.oversampling_amount = 5        # Number of times to oversample training images
options.be_verbose = True              # Display progress
options.random_seed = random.randint(0, 1000)  # Random seed for consistency

print("Starting model training with provided annotations...")

# Train the shape predictor
dlib.train_shape_predictor(xml_annotations, model_output_path, options)
print(f"Training complete. Model saved at: {model_output_path}")
