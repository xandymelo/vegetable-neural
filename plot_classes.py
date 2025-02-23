import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

class CustomVGG(nn.Module):
    def __init__(self, num_classes=15):
        super(CustomVGG, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256 * 56 * 56, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )
        
        self._initialize_weights()
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class ModelLoader:
    def __init__(self, model_path, num_classes=15):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path, num_classes)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _load_model(self, model_path, num_classes):
        model = CustomVGG(num_classes=num_classes)
        checkpoint = torch.load(model_path, map_location=self.device)
        
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model_dict = model.state_dict()
        
        filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict)

        for name, param in model.named_parameters():
            if name not in filtered_dict:
                print(f"Reinicializando {name}")
                if param.dim() >= 2:
                    nn.init.kaiming_normal_(param)
                else:
                    nn.init.constant_(param, 0)

        return model.to(self.device)
    
    def predict_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            max_prob, predicted = torch.max(probabilities, 0)
            
        return predicted.item(), max_prob.item(), probabilities.cpu().numpy()

def main():
    model_loader = ModelLoader(
        model_path="./models-trained/inception_model_config1.pth",
        num_classes=15
    )
    
    class_idx, prob, all_probs = model_loader.predict_image("dataset/vegetable-images/test/Bean/0001.jpg")
    print(f"Classe predita: {class_idx}, Probabilidade: {prob:.2f}")

if __name__ == "__main__":
    main()
