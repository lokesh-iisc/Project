# WGAN
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.ToTensor(),               
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

dataset = datasets.ImageFolder(root='animals_resized', transform=transform)
batch_size = 64
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
num_epochs = 200
z = 100


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 512 * 8 * 8),
            nn.BatchNorm1d(512 * 8 * 8),
            nn.ReLU(),
            nn.Unflatten(1, (512, 8, 8)),  # 8*8

            nn.ConvTranspose2d(512,256,4,2,1), #16*16
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.ConvTranspose2d(256,128,4,2,1), #32*32
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.ConvTranspose2d(128,64,4,2,1),#64*64
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64,32,3,1,1), #64*64
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32,3,4,2,1), #128*128
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)
    

class Discriminator(nn.Module):
    def __init__(self,):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16,4,2,1), 
            nn.LeakyReLU(0.2),  
            
            nn.Conv2d(16,32,4,2,1), 
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(32,64,4,2,1), 
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64,128,4,2,1),  
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Flatten(),  
            nn.Linear(128 * 8 * 8, 1), 
             
        )
    
    def forward(self, y):
        return self.model(y)

generator_w = Generator().to(device)
critic = Discriminator().to(device)

optimizer_g = optim.Adam(generator_w.parameters(), lr=0.0001, betas=(0.5, 0.999))
optimizer_d = optim.Adam(critic.parameters(), lr=0.0001, betas=(0.5, 0.999))

def gradient_penalty(critic, real_images, fake_images):
    batch_size = real_images.size(0)
    epsilon = torch.rand(batch_size, 1, 1, 1, device=device)  # Interpolation factor
    interpolated_images = epsilon * real_images + (1 - epsilon) * fake_images
    interpolated_images.requires_grad_(True)
    
    critic_interpolates = critic(interpolated_images)
    gradients = torch.autograd.grad(
        outputs=critic_interpolates, inputs=interpolated_images,
        grad_outputs=torch.ones_like(critic_interpolates), create_graph=True, retain_graph=True
    )[0]
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = ((gradient_norm - 1) ** 2).mean()
    return penalty


G_losses = []
D_losses = []

for epoch in range(num_epochs): 
    G_loss_epoch = 0.0
    D_loss_epoch = 0.0
    for batch in dataloader:
        real_imag = batch[0].to(device)
        batch_size = real_imag.size(0)
        for _ in range(5):
            optimizer_d.zero_grad()
            real_loss = torch.mean(critic(real_imag))
            x= torch.randn(batch_size,z).to(device)
            fake_imag = generator_w(x)
            fake_loss = torch.mean(critic(fake_imag.detach()))
            gp = gradient_penalty(critic, real_imag, fake_imag)
            d_loss = fake_loss -real_loss + 10 * gp
            d_loss.backward()
            optimizer_d.step()
        

        optimizer_g.zero_grad()
        fake_imag = generator_w(x)
        g_loss = -torch.mean(critic(fake_imag))
        g_loss.backward()
        optimizer_g.step()

        G_loss_epoch += g_loss.item()
        D_loss_epoch += d_loss.item()

    avg_G_loss = G_loss_epoch / len(dataloader)
    avg_D_loss = D_loss_epoch / len(dataloader)

    G_losses.append(avg_G_loss)
    D_losses.append(avg_D_loss)
    print(f"Epoch {epoch+1}/{num_epochs} | D Loss: {avg_D_loss:.4f} | G Loss: {avg_G_loss:.4f}")


generator_w.eval()

# Number of images in the grid
grid_size = 10
num_images = grid_size * grid_size

latent_vectors = torch.randn(num_images, 100).to(device)  # Assuming your latent vector size is 100

# Generate images with no gradient tracking
with torch.no_grad():
    generated_images = generator_w(latent_vectors).cpu()  # Generate images

# Denormalization function
def denormalize(image_tensor):
    image_tensor = (image_tensor * 0.5) + 0.5  # Normalize to [0, 1]
    return image_tensor.numpy()

# Denormalize the generated images
generated_images = denormalize(generated_images)

# Plotting the 10x10 grid of images
fig, axes = plt.subplots(grid_size, grid_size, figsize=(5, 5), squeeze=False)
for i in range(grid_size):
    for j in range(grid_size):
        img_index = i * grid_size + j
        axes[i, j].imshow(np.transpose(generated_images[img_index], (1, 2, 0)), aspect='auto')  # Rearranging dimensions for display
        axes[i, j].axis('off')  # Hide the axes

plt.subplots_adjust(wspace=0, hspace=0)  # Remove space between images
plt.tight_layout(pad=0)  # Further optimize layout
plt.show()

from torchvision import models, transforms
import random
from torchvision.transforms import ToPILImage
from scipy.linalg import sqrtm

class InceptionV3Features(nn.Module):
    def __init__(self):
        super(InceptionV3Features, self).__init__()
        self.inception = models.inception_v3(pretrained=True, transform_input=False)
        self.inception.fc = nn.Identity()  # Remove the final classification layer

    def forward(self, x):
        return self.inception(x)

inception_features_model = InceptionV3Features().to(device)
inception_features_model.eval()

transform = transforms.Compose([
    transforms.Resize((299, 299)),  # Resize images to 299x299
    transforms.ToTensor(),  # Convert images to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet statistics
])

indices = random.sample(range(len(dataset)), 300)

real_images = [dataset[i][0] for i in indices]
real_images = torch.stack(real_images)
print(real_images.shape)
to_pil = ToPILImage()
pil_images_real = [to_pil(real_images[i]) for i in range(real_images.size(0))]


x= torch.randn(300,100).to(device)
with torch.no_grad():  # Disable gradient calculation
    generated_images = generator_w(x)

def denormalize(image_tensor):
    image_tensor = (image_tensor * 0.5) + 0.5  # Scale from [-1, 1] to [0, 1]
    return image_tensor.clamp(0, 1)  

generated_images = denormalize(generated_images)
print(generated_images.shape)
pil_images_gen = [to_pil(generated_images [i]) for i in range(generated_images .size(0))]

def extract_features(pil_images):
    # Preprocess images (resize, normalize) and move to device
    images = torch.stack([transform(img) for img in pil_images]).to(device)  # Convert PIL images to tensors and move to GPU
    with torch.no_grad():  
        features = inception_features_model(images)  # Get 2048-dimensional features
    return features  

real_features = extract_features(pil_images_real)
print(real_features.shape)

generated_features = extract_features(pil_images_gen )

def fid(real_features,generated_features):    # d^2 = ||mu_1 – mu_2||^2 + Tr(C_1 + C_2 – 2*sqrt(C_1*C_2))
    mu_1 = np.mean(real_features, axis=0)
    mu_2 = np.mean(generated_features, axis=0)
    sigma_1 = np.cov(real_features, rowvar=False)
    sigma_2 = np.cov(generated_features, rowvar=False)

    a = np.sum((mu_1 - mu_2)**2)
    b = sqrtm(np.dot(sigma_1,sigma_2))

    if np.iscomplexobj(b):
        b = b.real

    fid = a + np.trace(sigma_1 + sigma_2 - 2*b)
    return fid

fid_score_wgan = fid(real_features.cpu().numpy(), generated_features.cpu().numpy())
print(f"FID Score: {fid_score_wgan}")

