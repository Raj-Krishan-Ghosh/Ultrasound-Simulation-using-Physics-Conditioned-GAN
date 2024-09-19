STEPS for using Ultrasound Simulation using Physics Conditioned GAN:

Step 1: Create a copy of the training code to your Google Drive by clicking on this link and then saving: https://drive.google.com/file/d/1Q2pZyqPnRq7yrar4a7dlC7pq9jXvoPLG/view?usp=sharing

Step 2: Open the Google Colab Notebook and ensure that you are using GPU by going to Edit -> Notebook Settings -> Hardware Accelerator -> T4 GPU

Step 3: Download the files from this repository (Dataloader.py, Dataset_BUSI.zip, Model.py, PolarPseudoBMode.py, weights.zip) and upload them to Google Colab Notebook Files

Step 4: Run the Google Colab Notebook cells one by one.


NOTES:

Note 1: The dataset used in this project is a toy dataset. The original dataset can be downloaded from https://scholar.cu.edu.eg/dataset_BUSI.zip for the actual training.

Note 2: The Pseudo B-Mode python code is a Python implementation of the matlab version https://www.mathworks.com/matlabcentral/fileexchange/34199-pseudo-b-mode-ultrasound-image-simulator

Note 3: The Pseudo B-Mode implementation method is based on the works of: Yongjian Yu, Acton, S.T., "Speckle reducing anisotropic diffusion," IEEE Trans. Image Processing, vol. 11, no. 11, pp. 1260-1270, Nov 2002.[http://dx.doi.org/10.1109/TIP.2002.804276] and J. C. Bambre and R. J. Dickinson, "Ultrasonic B-scanning: A computersimulation", Phys. Med. Biol., vol. 25, no. 3, pp. 463–479, 1980.[http://dx.doi.org/10.1088/0031-9155/25/3/006]

Note 4: Please give credit to the authors when using files from this repository. Using the files for anything other than learning purposes without credit to the authors is prohibited.
