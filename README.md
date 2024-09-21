# Steps for using Ultrasound Simulation using Physics Conditioned GAN

### Step 1
Create a copy of the training code to your Google Drive by clicking on this link and then saving: https://drive.google.com/file/d/1Q2pZyqPnRq7yrar4a7dlC7pq9jXvoPLG/view?usp=sharing

![ss01](https://github.com/user-attachments/assets/0dd4f0e6-07a6-405c-a8ee-b36b792dc7eb)

![ss02](https://github.com/user-attachments/assets/628513d4-fd1e-4f7a-94d6-afe0f72a77fb)

Step 2: Open the Google Colab Notebook and ensure that you are using GPU by going to Edit &#8594; Notebook Settings &#8594; Hardware Accelerator &#8594; T4 GPU

![ss03](https://github.com/user-attachments/assets/721318f3-dcd4-40b0-82ee-64b4cebc97fd)

![ss04](https://github.com/user-attachments/assets/d2a70078-2e0a-46a0-936d-2b26b91372d1)

Step 3: Download the files from this repository (Dataloader.py, Dataset_BUSI.zip, Model.py, PolarPseudoBMode.py, weights.zip) and upload them to Google Colab Notebook Files

![ss05](https://github.com/user-attachments/assets/dd457633-8d8f-4403-a96e-b7267fb6b215)

![ss06](https://github.com/user-attachments/assets/af8906da-7f69-47a2-ba8b-272256ed48e5)

Step 4: Run the Google Colab Notebook cells one by one.

![ss07](https://github.com/user-attachments/assets/d2e4a80d-8fa7-45c7-8e6a-7ac15c1036ad)

# Notes

Note 1: The dataset used in this project is a toy dataset for demonstration purposes only. The original dataset can be downloaded from https://scholar.cu.edu.eg/dataset_BUSI.zip for the actual training.

Note 2: The Pseudo B-Mode python code is a Python implementation of the matlab version https://www.mathworks.com/matlabcentral/fileexchange/34199-pseudo-b-mode-ultrasound-image-simulator

Note 3: The Pseudo B-Mode implementation method is based on the works of: Yongjian Yu, Acton, S.T., "Speckle reducing anisotropic diffusion," IEEE Trans. Image Processing, vol. 11, no. 11, pp. 1260-1270, Nov 2002.[http://dx.doi.org/10.1109/TIP.2002.804276] and J. C. Bambre and R. J. Dickinson, "Ultrasonic B-scanning: A computersimulation", Phys. Med. Biol., vol. 25, no. 3, pp. 463–479, 1980.[http://dx.doi.org/10.1088/0031-9155/25/3/006]
