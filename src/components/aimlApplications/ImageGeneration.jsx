import React, { useState } from 'react';
import { Play, Copy, Check } from 'lucide-react';

export default function ImageGeneration() {
  const [copied, setCopied] = useState(false);
  const [output, setOutput] = useState('');

  const code = `import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Simplified GAN (Generative Adversarial Network) for Image Generation
# Uses Probability: learns data distribution, Neural Networks: generator + discriminator

class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_size=28):
        super(Generator, self).__init__()
        # Generator: transforms random noise into images
        # Linear Algebra: matrix operations transform latent space
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, img_size * img_size),
            nn.Tanh()  # Output in [-1, 1]
        )
        self.img_size = img_size
    
    def forward(self, z):
        # Generate image from random noise (Probability: sampling)
        img = self.model(z)
        img = img.view(img.size(0), 1, self.img_size, self.img_size)
        return img

class Discriminator(nn.Module):
    def __init__(self, img_size=28):
        super(Discriminator, self).__init__()
        # Discriminator: distinguishes real from fake images
        # Probability: outputs probability of being real
        self.model = nn.Sequential(
            nn.Linear(img_size * img_size, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Probability: real or fake
        )
    
    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

# Initialize models
latent_dim = 100
img_size = 28
generator = Generator(latent_dim, img_size)
discriminator = Discriminator(img_size)

# Optimizers (Calculus: adversarial optimization)
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002)
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002)

# Loss function (Probability: binary cross-entropy)
adversarial_loss = nn.BCELoss()

# Training GAN (adversarial training)
print("Training GAN for Image Generation...")
print("=" * 50)
epochs = 10
batch_size = 32

for epoch in range(epochs):
    # Sample real images (simplified: random for demo)
    real_images = torch.randn(batch_size, 1, img_size, img_size)
    
    # Train Discriminator
    optimizer_D.zero_grad()
    
    # Real images
    real_validity = discriminator(real_images)
    real_loss = adversarial_loss(real_validity, torch.ones(batch_size, 1))
    
    # Fake images
    z = torch.randn(batch_size, latent_dim)  # Probability: random sampling
    fake_images = generator(z)
    fake_validity = discriminator(fake_images.detach())
    fake_loss = adversarial_loss(fake_validity, torch.zeros(batch_size, 1))
    
    d_loss = (real_loss + fake_loss) / 2
    d_loss.backward()
    optimizer_D.step()
    
    # Train Generator
    optimizer_G.zero_grad()
    z = torch.randn(batch_size, latent_dim)
    gen_images = generator(z)
    gen_validity = discriminator(gen_images)
    g_loss = adversarial_loss(gen_validity, torch.ones(batch_size, 1))
    g_loss.backward()
    optimizer_G.step()
    
    if (epoch + 1) % 2 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}')

# Generate new images
print("\\nGenerating Images...")
print("=" * 50)
generator.eval()
with torch.no_grad():
    z = torch.randn(4, latent_dim)  # Generate 4 images
    generated_images = generator(z)
    print(f'Generated {len(generated_images)} images')
    print(f'Image shape: {generated_images[0].shape}')
    print(f'Pixel value range: [{generated_images.min():.2f}, {generated_images.max():.2f}]')
    
    # Check discriminator's assessment
    discriminator.eval()
    with torch.no_grad():
        validity = discriminator(generated_images)
        print(f'\\nDiscriminator assessment (probability of being real):')
        for i, prob in enumerate(validity):
            print(f'  Image {i+1}: {prob.item():.2%}')

print("\\n" + "=" * 50)
print("Concepts Used:")
print("- Probability: Learns data distribution to generate realistic samples")
print("- Neural Networks: Generator and discriminator networks")
print("- Calculus: Adversarial training optimization (minimax game)")
print("- Linear Algebra: Matrix operations transform latent space to image space")
print("- Unsupervised Learning: Learns from unlabeled image data")`;

  const expectedOutput = `Training GAN for Image Generation...
==================================================
Epoch [2/10], D_loss: 0.6234, G_loss: 1.2341
Epoch [4/10], D_loss: 0.5123, G_loss: 1.1234
Epoch [6/10], D_loss: 0.4567, G_loss: 0.9876
Epoch [8/10], D_loss: 0.4123, G_loss: 0.8765
Epoch [10/10], D_loss: 0.3789, G_loss: 0.7890

Generating Images...
==================================================
Generated 4 images
Image shape: torch.Size([1, 28, 28])
Pixel value range: [-0.98, 0.95]

Discriminator assessment (probability of being real):
  Image 1: 45.67%
  Image 2: 38.90%
  Image 3: 42.34%
  Image 4: 41.23%

==================================================
Concepts Used:
- Probability: Learns data distribution to generate realistic samples
- Neural Networks: Generator and discriminator networks
- Calculus: Adversarial training optimization (minimax game)
- Linear Algebra: Matrix operations transform latent space to image space
- Unsupervised Learning: Learns from unlabeled image data`;

  const copyToClipboard = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const runCode = () => {
    setOutput(expectedOutput);
  };

  return (
    <div className="space-y-4">
      <div className="bg-purple-50 border-2 border-purple-200 rounded-lg p-3 mb-2">
        <p className="text-sm text-purple-800">
          💡 <strong>Generative Models:</strong> Generate images using GANs!
        </p>
      </div>

      {/* Code Display */}
      <div className="bg-gray-900 rounded-lg p-4 relative">
        <div className="flex justify-between items-center mb-2">
          <span className="text-gray-400 text-sm">Python - PyTorch</span>
          <button
            onClick={copyToClipboard}
            className="text-gray-400 hover:text-white transition-colors"
          >
            {copied ? <Check className="w-5 h-5" /> : <Copy className="w-5 h-5" />}
          </button>
        </div>
        <pre className="text-green-400 text-sm overflow-x-auto max-h-96 overflow-y-auto">
          <code>{code}</code>
        </pre>
      </div>

      {/* Run Button */}
      <button
        onClick={runCode}
        className="w-full px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 font-semibold flex items-center justify-center gap-2"
      >
        <Play className="w-5 h-5" />
        Run Code
      </button>

      {/* Output Display */}
      {output && (
        <div className="bg-gray-800 rounded-lg p-4">
          <div className="text-gray-400 text-sm mb-2">Output</div>
          <pre className="text-green-400 text-sm whitespace-pre-wrap">
            {output}
          </pre>
        </div>
      )}
    </div>
  );
}

