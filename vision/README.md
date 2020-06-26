* record_demos.py is used to collect the high-dimensional action demonstrations
* process_demos.py converts those demonstrations into a dataset of state-action pairs
* train_model.py learns a conditional autoencoder from the dataset
* teleop_cae.py lets the user control the robot with the latent space

This task is something from our RSS paper.
The robot has been shown demonstrations reaching for the orange box.
It reaches in two different styles: from the side, and from the top.
Your latent input controls which style it 'grabs' with.
The blue box is an artifact meant to confuse the robot.
