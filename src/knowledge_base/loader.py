from datasets import load_dataset
from loguru import logger

class DrugsDataset:
    def __init__(self,
                 train_dataset,
                 test_dataset,
                 ):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.dataset_dict = {
            "train": self.train_dataset,
            "test": self.test_dataset
        }
        self.dataset = self.initialize()
        
        self.num_train = len(self.dataset["train"])
        self.num_test = len(self.dataset["test"])
        
    def initialize(self):
        return load_dataset("csv", 
                            data_files=self.dataset_dict, 
                            delimiter="\t")
        
    def preprocess(self):
        # Rename id column
        try:
            self.dataset = self.dataset.rename_column("Unnamed: 0", "patient_id")
        except:
            logger.warning("Column 'Unnamed: 0' not found.")

        # Remove rows with missing values
        self.dataset = self.dataset.dropna()
        
        
        
         
        
    
        