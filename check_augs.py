import torch

normal_path = "../mount_outputs/features/treatment_Q90_hipt4096_features_normalised_updatedsegmentation/pt_files/2-1613704B.pt"
aug_path = "../mount_outputs/features/treatment_Q90_hipt4096_features_normalised_updatedsegmentation/pt_files/2-1613704Baug1.pt"
aug2_path = "../mount_outputs/features/treatment_Q90_hipt4096_features_normalised_updatedsegmentation/pt_files/2-1613704Baug2.pt"
aug3_path = "../mount_outputs/features/treatment_Q90_hipt4096_features_normalised_updatedsegmentation/pt_files/2-1613704Baug3.pt"
#normal_features = torch.load(normal_path)
#aug_features = torch.load(aug_path)
#aug2_features = torch.load(aug2_path)
#aug3_features = torch.load(aug3_path)
print(torch.load(normal_path))
print("AUG1 FEATURES: \n") 
print( torch.load(aug_path))
print("AUG2 FEATURES: \n")
print(torch.load(aug2_path))
print("AUG3 FEATURES: \n")
print(torch.load(aug3_path))
