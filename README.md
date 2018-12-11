# Final-Project-Group5
Constructing Convolutional Neural Networks to train on the RVL-CDIP dataset

Image Sorting, EDA, and Preprocessing:
      - ExtractionAndSorting.ipynb
          - Extracts file paths and labels from labels textfile and moves to new DocImages folder
          - Change directory as needed
      - EDA.ipynb
          - Extracts image dimensions distribution, pixel mean value, max and min pixel images, and increasingly small document images
          - Change directories to where the images are stored
      - Preprocessing.ipynb
          - Checks preprocessing methods on random images from each class
          - Histogram Equalization
          - Creates new dataset for each image region (will take long time due to number of images being created)
      - labels folder contains the original label textfiles


CNN Models

PyTorch:
- MakeCSV.py: 
      - Run on training and testing image data before CNN. This will make a csv file
	      with image paths and labels
- CNN_Pytorch.py: 
      - Link to newly created csv below line: "if __name__ == '__main__':" to load images
      - Change model name after testing to specify save location
- All models run on a Google Cloud instance


Plotting Metrics:

- Plotting_Metric_Graphs.ipynb
      - Plots bar graphs of metric scores
      - CSV's are read in and transformed to be suitable for seaborn graphing
      - Model metric csv's are created manually from the outputs of CNN_PyTorch.py in the structure of the following table:
      
      Learning Rate	Accuracy	Recall	Precision	     F1	Training Time
               0.01	   0.606	  0.61	    0.633	  0.603	         3046
              0.001	   0.728	 0.729	    0.736	  0.723	         3050
             0.0001	   0.705	 0.708	    0.717	  0.701	         3039
            0.00001	   0.611	 0.615	    0.608	  0.608          3063

- Plotting_Loss.ipynb
      - Plots are line graphs of loss over the iterations
      - CSV's are a single row of consecutive losses. A list of losses is generated in CNN_PyTorch.py. Run the command print(losses) after the
        model is created to get a list of losses. Copy and paste this list into Notepad, manually remove the opening and closing brackets,
        and save as a csv.
        
        It should look like this:
        
        2.803864956, 10.76157856, 6.767965317, 4.519660473, 3.508453608, 5.588068962, 3.565412283, 4.532711506, 3.379324675............	
        
- Plotting_CM.ipynb
      - Plots the confusion matrix of the final model.
      - The confusion matrix data is copied directly from the output of the confusion_matrix variable from CNN_PyTorch.py and pasted
        into the cm variable. Commas are added by hand to make it a nested list.
        
- Plotting_Kernels.ipynb
      - Plots all 32 kernels of the 1st layer of a model
      - The kernel csv's are made from CNN_PyTorch.py and downloaded from GCP into the Kernels folder. The csv's are 32 rows (one for each
        kernel) and n columns (n = number of weight values in the kernel).
      - Each kernel row is reshaped into a square kernel.
      
- PlottingFeatureMap.ipynb
      - Plots the output feature maps of each convolution/pooling block
      - CSV's of feature map values are extracted and saved form CNN_PyTorch.py using PyTorch's debugger and downloaded from GCP
      - Each csv consists of an index and 1 column and n rows (n = number of weight values in feature map). 
      - The values column is subsetted and reshaped into squares 

