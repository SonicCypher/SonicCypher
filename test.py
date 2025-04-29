# import os
# print(os.listdir("./logdir/"))

# import tensorflow as tf

# # Path to your checkpoint (without file extensions)
# checkpoint_path = "./logdir/ref_encoder.ckpt-5000"

# # Create a checkpoint reader
# reader = tf.train.NewCheckpointReader(checkpoint_path)

# # Get a list of all variables in the checkpoint
# all_variables = reader.get_variable_to_shape_map()

# # Print variable names and their shapes
# for var_name, shape in all_variables.items():
#     print(f"Variable: {var_name}, Shape: {shape}")


import tensorflow as tf

checkpoint_path = './logdir/ref_encoder_updated.ckpt'
checkpoint = tf.train.load_checkpoint(checkpoint_path)

# List all variables in the checkpoint
var_names = checkpoint.get_variable_to_shape_map()
#print("Variables in checkpoint:", var_names)

from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

print_tensors_in_checkpoint_file(file_name=checkpoint_path, tensor_name='', all_tensors=False)
