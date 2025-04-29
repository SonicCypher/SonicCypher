import tensorflow as tf
import os

def update_checkpoint_with_new_names(old_checkpoint_path, new_checkpoint_path):
    # Check if the new checkpoint file already exists, if not, create a new checkpoint
    if not os.path.exists(new_checkpoint_path):
        print(f"Creating a new checkpoint at: {new_checkpoint_path}")
    else:
        print(f"Checkpoint {new_checkpoint_path} already exists, it will be overwritten.")
    
    # Load the original checkpoint
    reader = tf.train.load_checkpoint(old_checkpoint_path)
    variable_map = {}

    # Gather variables from the original checkpoint
    variables_in_checkpoint = reader.get_variable_to_shape_map().items()

    for name, shape in variables_in_checkpoint:
        # Adjust the variable name by removing 'net/' prefix from checkpoint
        if name.startswith('net/reference_encoder/'):
            new_name = name.replace('net/reference_encoder/', 'reference_encoder/')
            print(f"Mapping: {name} -> {new_name}")
            variable_map[new_name] = name

    # Create a new checkpoint to save the updated variable names
    with tf.Session() as sess:
        # Create new variables with the updated names
        new_variables = {}
        for new_name, original_name in variable_map.items():
            # Convert tensor into a variable
            new_variable = tf.Variable(reader.get_tensor(original_name), name=new_name)
            new_variables[new_name] = new_variable
        
        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        
        # Restore the variables into the session
        saver = tf.train.Saver(var_list=new_variables)
        saver.save(sess, new_checkpoint_path)
    
    print(f"Checkpoint saved with updated variable names at: {new_checkpoint_path}")

# Usage
old_checkpoint_path = './logdir/ref_encoder.ckpt-5000'
new_checkpoint_path = './logdir/ref_encoder_updated.ckpt'

# Ensure the new checkpoint directory exists before starting
os.makedirs(os.path.dirname(new_checkpoint_path), exist_ok=True)

# Start updating and saving the checkpoint
update_checkpoint_with_new_names(old_checkpoint_path, new_checkpoint_path)
