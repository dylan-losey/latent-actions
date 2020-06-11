* record_demos.py is used to collect the high-dimensional action demonstrations
* process_demos.py converts those demonstrations into a dataset of state-action pairs
* train_model.py learns a conditional autoencoder from the dataset
* teleop_cae.py lets the user control the robot with the latent space

The current task is moving the end-effector of the Panda robot in a plane.
