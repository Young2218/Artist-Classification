from torch.utils.data import Dataset
import cv2

class VitDataset(Dataset):
    def __init__(self, img_paths, labels, feature_extractor,transforms=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transforms = transforms
        self.feature_extractor = feature_extractor

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transforms is not None:
            image = self.transforms(image=image)['image']
        
        image_feature = self.feature_extractor(images=image, return_tensors="pt")
        
        if self.labels is not None:
            label = self.labels[index]
            return image_feature['pixel_values'][0], label
        else:
            return image_feature['pixel_values'][0]
    
    def __len__(self):
        return len(self.img_paths)