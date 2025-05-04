import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import TensorDataset, DataLoader
from base import BaseCustom
from sklearn.base import ClassifierMixin
from sklearn.calibration import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from steps.deep_learning.spectrogram_augmenter import SpectrogramAugmenter
from steps.deep_learning.optimizer import catalog as optim_catalog
from steps.deep_learning.loss_function import catalog as loss_catalog


class DLModelWrapper(BaseCustom, ClassifierMixin):
    """
    Wrapper for Deep Learning models for sklearn compatibility. Supports catalog-based
    model, optimizer and loss function instantiation as well as configurable spectrogram
    augmentation

    ** Params
    - num_classes: Number of classes of the data. Defaults to 4
    - epochs: number of training epochs. Defaults to 10
    - batch_size: Size of the training batches to apply in each epoch. Defaults to 32
    - learning_rate: Training learning rate. Defaults to 1e-3
    - model_name: Name of the network architecture as defined in the models catalog. Defaults to "resnet18"
    - optimizer_name: Name of the optimizer as defined in the catalog. Defaults to "adam"
    - optimizer_params: Dictionary with the parameters of the optimizer. If its a default PyTorch
                        optimizer names match the expected by its constructor. None by default, to use
                        optimizer's default parameters
    - loss_name: Name of the loss function as defined in the catalog
    - loss_params: Dictionary with the parameters of the loss function. If its a default PyTorch
                   function names match the expected by its constructor. None by default, to use
                   loss function's default parameters
    - augmentation_config: Dictionary with the configuration for spectrogram augmentation. None by default,
                           indicating not to perform augmentation
    - class_weights: Same inputs as sklearn's compute_class_weight. Defaults to "balanced"
    - device: Device name to execute PyTorch processing. None by default, meaning it will try to use CUDA or fallback to CPU
    """
    def __init__(self, 
                 num_classes=4, 
                 epochs=10, 
                 batch_size=32, 
                 learning_rate=1e-3, 
                 model_name="resnet18",
                 optimizer_name="adam",
                 optimizer_params={},
                 loss_name="crossentropy",
                 loss_params={},
                 augmentation_config=None,
                 class_weights="balanced",
                 device=None):
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model_name = model_name
        self.optimizer_name = optimizer_name
        self.optimizer_params = optimizer_params
        self.loss_name = loss_name
        self.loss_params = loss_params
        self.augmentation_config = augmentation_config
        self.class_weights = class_weights  # Optional, can be auto-computed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.label_encoder = None

    def fit(self, X, y):
        self.label_encoder = LabelEncoder()
        y = torch.tensor(self.label_encoder.fit_transform(y), dtype=torch.long)

        dataset = TensorDataset(torch.stack(X), y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Initialize model and optimizer
        self.model = catalog[self.model_name](self.num_classes).to(self.device)
        augment = SpectrogramAugmenter(**self.augmentation_config) if self.augmentation_config else None
        optimizer =  optim_catalog[self.optimizer_name](self.model.updatable_params, lr=self.learning_rate)
        loss_type, supports_weights = loss_catalog[self.loss_name]

        # Weights definition when supported and configured
        if supports_weights and self.class_weights:
            if isinstance(self.class_weights, dict):
                weights_input = {self.label_encoder.transform([cls_name])[0]: weight for cls_name, weight in self.class_weights.items()}
            else:
                weights_input = self.class_weights

            classes = np.unique(y.cpu().numpy())
            class_weights_array = compute_class_weight(class_weight=weights_input, classes=classes, y=y.cpu().numpy())
            class_weights = torch.tensor(class_weights_array, dtype=torch.float32)  
            criterion = loss_type(weight=class_weights.to(self.device))

        else:
            criterion = loss_type(**self.loss_params)

        # Set model to train mode and perform training loop
        self.model.train()

        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                if augment:
                    batch_x = augment(batch_x)

                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        return self

    def predict(self, X):
        loader = DataLoader(torch.stack(X), batch_size=self.batch_size)

        self.model.eval()
        preds = []

        with torch.no_grad():
            for batch_x in loader:
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x)
                batch_preds = outputs.argmax(dim=1).cpu().numpy()
                preds.extend(batch_preds)

        return self.label_encoder.inverse_transform(preds)

    def predict_proba(self, X):
        loader = DataLoader(torch.stack(X), batch_size=self.batch_size)

        self.model.eval()
        probs = []

        with torch.no_grad():
            for batch_x in loader:
                batch_x = batch_x.to(self.device)
                outputs = F.softmax(self.model(batch_x), dim=1)
                probs.append(outputs.cpu().numpy())

        return np.vstack(probs)


class ResNet18(nn.Module):
    """
    ResNet18 wrapper to optionally adapt the architecture to a one-channel input and the specified number of classes

    ** Params:
    - num_classes: Number of classes of the data. Defaults to 4
    - input_channels: Number of channels of the input layer. When not 1 then only the output layer will be adapted. Defaults to 1 
    """
    def __init__(self, num_classes=4, input_channels=1):
        super().__init__()
        self.model = models.resnet18(pretrained=True)

        for parameter in self.model.parameters():
            parameter.requires_grad = False

        if input_channels == 1:
            self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        # Step 3: Update final classification layer
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        self.updatable_params = [param for _, param in self.model.named_parameters() if param.requires_grad] 

    def forward(self, x):
        return self.model(x)


catalog = {
    "resnet18": ResNet18
}