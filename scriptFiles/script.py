class DeepfakeDataset(Dataset):
    def __init__(self, videos, labels, processor, frame_count=5, transform=None):
        self.videos = videos
        self.labels = labels
        self.processor = processor
        self.frame_count = frame_count
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_path = self.videos[idx]
        label = self.labels[idx]


    # Extract frames from video
        cap = cv2.VideoCapture(video_path)
        frames = []
        for _ in range(self.frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frames.append(self.transform(frame))
        cap.release()

        # Handle empty frames
        if len(frames) == 0:
            # Add default blank frames of size [3, 224, 224]
            blank_frame = torch.zeros(3, 224, 224)  # RGB with height and width
            frames = [blank_frame] * self.frame_count
    # Pad frames if less than required
        while len(frames) < self.frame_count:
            frames.append(torch.zeros_like(frames[0]))

        # Stack frames into a tensor and aggregate
        frames_tensor = torch.stack(frames)
        aggregated_frame = frames_tensor.mean(dim=0)

        # Ensure the pixel values are within [0, 255]
        aggregated_frame = aggregated_frame * 255  # Scale to [0, 255]
        aggregated_frame = aggregated_frame.clamp(0, 255).byte()  # Convert to uint8

        # Process the aggregated frame using the processor
        inputs = self.processor(images=aggregated_frame, return_tensors="pt", do_rescale=False)
        pixel_values = inputs['pixel_values'].squeeze(0)

        return pixel_values, torch.tensor(label)

# Hyperparameter tuning (important for maximizing accuracy)
BATCH_SIZE = 32  # Adjust based on GPU memory
LEARNING_RATE = 1e-5  # Start with a small learning rate
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 10  # Adjust as needed
IMAGE_SIZE = 224 # Adjust as needed. MobileViT can work with different sizes.

# Initialize Dataset and DataLoader
processor = AutoImageProcessor.from_pretrained("apple/mobilevit-small")  # Or a smaller one

# Use a transform with more augmentations (important for generalization)
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # Use the new image size
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet stats
])

val_transform = transforms.Compose([ # Validation transform (no random augmentations)
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), # Use the new image size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet stats
])

train_dataset = DeepfakeDataset(train_videos, train_labels, processor, transform=train_transform) # Pass transform
val_dataset = DeepfakeDataset(val_videos, val_labels, processor, transform=val_transform) # Pass transform


train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


# Model Setup (MobileViT - Try different sizes)
# Option 1: Pre-trained MobileViT from Hugging Face
model = MobileViTForImageClassification.from_pretrained("apple/mobilevit-small")  # Or "apple/mobilevit-xx-small"
# Option 2: More variants from timm (install with: pip install timm)
# model = timm.create_model('mobilevit_xx_small', pretrained=True, num_classes=2) # Example. See timm docs for all models

model.config.num_labels = 2
model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))


# Training Loop (with early stopping and saving best model)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
criterion = torch.nn.CrossEntropyLoss()
device = model.device

best_val_acc = 0.0
patience = 3  # For early stopping
patience_counter = 0

# ... (evaluate function - same as before)

for epoch in range(NUM_EPOCHS):
    # ... (training loop - same as before)

    val_loss, val_accuracy = evaluate(model, val_loader, criterion)

    print(f"Epoch {epoch+1}: Train Loss = ..., Train Acc = ..., Val Loss = {val_loss:.4f}, Val Acc = {val_accuracy:.4f}")

    # Early stopping and best model saving
    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        patience_counter = 0
        torch.save(model.state_dict(), "best_mobilevit_model.pth") # Save best model
        print("Best model saved!")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping!")
            break  # Exit the training loop



# Load and evaluate the best model
best_model = MobileViTForImageClassification.from_pretrained("apple/mobilevit-small") # Or timm if you used it.
best_model.load_state_dict(torch.load("best_mobilevit_model.pth"))
best_model.to(device)

# ... (evaluate best_model on your test set)
