# ChromeTrex-ML
We used supervised learning and developed a reinforcement learning approach to learn the Chrome TRex game at very high speeds that a human player would find difficult. 

This gif is the final product: an actual run in the advanced difficulty mode of the game generated from the best predictions from the highest performing ML algorithm. The highest accuracy achieved was a 97% obstacle clearance rate at the normal game speed and 89.8% at very high speeds. The video is not sped up. 

![Alt Text](https://github.com/OmkarSavant/ChromeTrex-ML/blob/master/sampleRun.gif)

A paper describing this project is here: https://www.dropbox.com/s/2fv6wsfxtymz5fr/Team%2022%20Final%20Paper.pdf?dl=0

The paper explains the problem we were attempting to solve, feature representations tested, the various ML algorithms that were tested, and outlines a pseudocode for the reinforcement learning approach. 

The file 'main_auto_new_multiple.py' is an automated version of the Chrome game where the dinosaur makes random guesses for when to jump over an obstacle while running at random speeds, and obstacles of random sizes are generated. All examples of successful jumps are saved to a file with information about the environment and the distance the dinosaur jumped at. These are used for learning. 

The file 'algos.py' pre-processes this data for use in a variety of ML algorithms, and each cell runs a different type of model. Then, test data is generated, and predictions are made. These test obstacles and speed, along with their corresponding predictions for jump distances are saved to a file. 

The file 'models_dino_multiple.py' extracts the predictions and the corresponding obstacles and speeds and runs the game for 250 obstacles. This will yield the accuracy of the model that was used to generate the predictions. 

* The original pygame version of the game is by Shivam Shekhar: https://github.com/shivamshekhar/Chrome-T-Rex-Rush * We modified it to remove pterodactyles, generate exapmles of good jumps and save this to a spreadsheet, customize random obstacles and speeds, run the testing from the ML predictions, and add additional gameplay information to the display.
