# DeepLearning 2024 Project

## Milestone 2

### Tasks Regarding Milestone 2:
- We are trying to improve the performance of the model by addig augmentation to the images.
- We trained models separetly with different parameters, to start ensembling models.

### Instruction to Run Files:
- `train_model.py`: The images must be in the specified folder in `X_.nii` format
- `model_test_vis.ipynb`: You can visualise the predicted masks, of the different models, what can loadad from .pt files.
- `model.pt`: Contain the saved pytorch model
- `U-net_high_queality.png` shows the model achitecture
- `model_test.png` shows the output of the different models, you can downloade them from this link 

### Future plans:
- Add learning rate scheduler nad early stoppint to into the training function
- Create Ensemble model from the best performing models.
