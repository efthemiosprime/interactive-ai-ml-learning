import React, { useState } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

export default function PretrainedModels() {
  const [selectedFramework, setSelectedFramework] = useState('pytorch');
  const [selectedModel, setSelectedModel] = useState('bert');

  const pytorchCode = {
    bert: `import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from transformers import pipeline

# 1. Using Pre-trained BERT for Text Classification
def use_bert_classifier():
    """Use pre-trained BERT for sentiment analysis"""
    # Load pre-trained model and tokenizer
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # Example text
    text = "I love this product! It's amazing."
    
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # Get label
    label_id = predictions.argmax().item()
    confidence = predictions[0][label_id].item()
    
    labels = ["NEGATIVE", "POSITIVE"]
    print(f"Label: {labels[label_id]}, Confidence: {confidence:.4f}")
    
    return labels[label_id], confidence

# 2. Using Hugging Face Pipeline (Easiest Way)
def use_pipeline():
    """Use Hugging Face pipeline for easy inference"""
    # Sentiment Analysis
    classifier = pipeline("sentiment-analysis")
    result = classifier("I love using pre-trained models!")
    print(result)  # [{'label': 'POSITIVE', 'score': 0.9998}]
    
    # Text Generation
    generator = pipeline("text-generation", model="gpt2")
    result = generator("The future of AI is", max_length=50, num_return_sequences=1)
    print(result)
    
    # Question Answering
    qa_pipeline = pipeline("question-answering")
    result = qa_pipeline(
        question="What is machine learning?",
        context="Machine learning is a subset of artificial intelligence..."
    )
    print(result)
    
    # Named Entity Recognition
    ner_pipeline = pipeline("ner", aggregation_strategy="simple")
    result = ner_pipeline("Apple is headquartered in Cupertino, California.")
    print(result)

# 3. Using Pre-trained Vision Models (ResNet, EfficientNet)
import torchvision.models as models
from torchvision import transforms
from PIL import Image

def use_resnet_classifier():
    """Use pre-trained ResNet for image classification"""
    # Load pre-trained ResNet50
    model = models.resnet50(pretrained=True)
    model.eval()
    
    # Image preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess image
    image = Image.open("image.jpg")
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    
    # Get predictions
    with torch.no_grad():
        output = model(input_batch)
    
    # Get top 5 predictions
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top5_prob, top5_idx = torch.topk(probabilities, 5)
    
    # Load ImageNet labels
    import json
    with open("imagenet_labels.json") as f:
        labels = json.load(f)
    
    for i in range(5):
        print(f"{labels[top5_idx[i]]}: {top5_prob[i].item():.4f}")

# 4. Using Pre-trained GPT Models
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def use_gpt2():
    """Use pre-trained GPT-2 for text generation"""
    # Load model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    
    # Set pad token
    tokenizer.pad_token = tokenizer.eos_token
    
    # Input text
    prompt = "The future of artificial intelligence"
    
    # Tokenize
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    # Generate
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=100,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(generated_text)

# 5. Using Pre-trained YOLO for Object Detection
from ultralytics import YOLO

def use_yolo():
    """Use pre-trained YOLO for object detection"""
    # Load pre-trained YOLOv8
    model = YOLO("yolov8n.pt")  # nano version
    
    # Run inference
    results = model("image.jpg")
    
    # Process results
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = box.conf[0].cpu().item()
            class_id = int(box.cls[0].cpu().item())
            class_name = model.names[class_id]
            
            print(f"{class_name}: {confidence:.2f} at [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")

# 6. Using Pre-trained CLIP for Image-Text Matching
from transformers import CLIPProcessor, CLIPModel

def use_clip():
    """Use pre-trained CLIP for image-text matching"""
    # Load model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Image and text
    image = Image.open("image.jpg")
    texts = ["a cat", "a dog", "a bird"]
    
    # Process inputs
    inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
    
    # Get embeddings
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
    
    # Get best match
    best_match_idx = probs.argmax().item()
    print(f"Best match: {texts[best_match_idx]} ({probs[0][best_match_idx]:.4f})")

# Example Usage
if __name__ == "__main__":
    # BERT for classification
    use_bert_classifier()
    
    # Hugging Face pipeline (easiest)
    use_pipeline()
    
    # ResNet for image classification
    # use_resnet_classifier()
    
    # GPT-2 for text generation
    use_gpt2()`,

    resnet: `import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import json

# 1. Load Pre-trained ResNet Models
def load_resnet_models():
    """Load different ResNet architectures"""
    # ResNet18 (smaller, faster)
    resnet18 = models.resnet18(pretrained=True)
    
    # ResNet50 (balanced)
    resnet50 = models.resnet50(pretrained=True)
    
    # ResNet101 (larger, more accurate)
    resnet101 = models.resnet101(pretrained=True)
    
    # ResNet152 (largest)
    resnet152 = models.resnet152(pretrained=True)
    
    return resnet50  # Return ResNet50 as default

# 2. Image Preprocessing for ImageNet
def preprocess_image(image_path):
    """Preprocess image for ImageNet models"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor

# 3. Image Classification
def classify_image(model, image_path):
    """Classify image using pre-trained ResNet"""
    model.eval()
    
    # Preprocess image
    input_tensor = preprocess_image(image_path)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    
    # Get top 5 predictions
    top5_prob, top5_idx = torch.topk(probabilities, 5)
    
    # Load ImageNet labels
    with open("imagenet_labels.json") as f:
        labels = json.load(f)
    
    results = []
    for i in range(5):
        results.append({
            "label": labels[top5_idx[i]],
            "confidence": top5_prob[i].item()
        })
        print(f"{labels[top5_idx[i]]}: {top5_prob[i].item():.4f}")
    
    return results

# 4. Feature Extraction
def extract_features(model, image_path):
    """Extract features from image using ResNet"""
    # Remove final classification layer
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()
    
    # Preprocess image
    input_tensor = preprocess_image(image_path)
    
    # Extract features
    with torch.no_grad():
        features = model(input_tensor)
        features = features.squeeze()
    
    print(f"Feature vector shape: {features.shape}")
    return features

# 5. Batch Processing
def classify_batch(model, image_paths):
    """Classify multiple images"""
    model.eval()
    
    # Preprocess all images
    images = [preprocess_image(path) for path in image_paths]
    batch = torch.cat(images, dim=0)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(batch)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    
    # Get top predictions for each image
    top_probs, top_indices = torch.topk(probabilities, 1, dim=1)
    
    return top_indices.squeeze().tolist(), top_probs.squeeze().tolist()

# Example Usage
if __name__ == "__main__":
    # Load model
    model = load_resnet_models()
    
    # Classify single image
    results = classify_image(model, "image.jpg")
    
    # Extract features
    features = extract_features(model, "image.jpg")
    
    # Batch processing
    image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
    predictions, confidences = classify_batch(model, image_paths)`,

    gpt: `import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Using GPT-2 (Smaller, Faster)
def use_gpt2():
    """Use pre-trained GPT-2 for text generation"""
    # Load model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    
    # Set pad token
    tokenizer.pad_token = tokenizer.eos_token
    
    # Input prompt
    prompt = "The future of artificial intelligence"
    
    # Tokenize
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    # Generate text
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=150,
            num_return_sequences=3,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode generated text
    for i, output in enumerate(outputs):
        text = tokenizer.decode(output, skip_special_tokens=True)
        print(f"Generated Text {i+1}:\\n{text}\\n")

# 2. Using GPT-2 with Custom Parameters
def generate_with_params(prompt, max_length=100, temperature=0.7, top_k=50):
    """Generate text with custom parameters"""
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 3. Using Larger GPT Models (GPT-2 Medium, Large, XL)
def use_larger_gpt():
    """Use larger GPT-2 variants"""
    models = {
        "gpt2": "gpt2",  # 124M parameters
        "gpt2-medium": "gpt2-medium",  # 355M parameters
        "gpt2-large": "gpt2-large",  # 774M parameters
        "gpt2-xl": "gpt2-xl"  # 1.5B parameters
    }
    
    # Use GPT-2 Medium
    tokenizer = GPT2Tokenizer.from_pretrained(models["gpt2-medium"])
    model = GPT2LMHeadModel.from_pretrained(models["gpt2-medium"])
    tokenizer.pad_token = tokenizer.eos_token
    
    prompt = "In the world of machine learning"
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=200,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# 4. Using GPT Models via Hugging Face Pipeline
from transformers import pipeline

def use_gpt_pipeline():
    """Use GPT via pipeline (easiest way)"""
    generator = pipeline(
        "text-generation",
        model="gpt2",
        device=0 if torch.cuda.is_available() else -1
    )
    
    prompt = "The impact of AI on society"
    result = generator(
        prompt,
        max_length=100,
        num_return_sequences=1,
        temperature=0.7
    )
    
    print(result[0]["generated_text"])

# 5. Chat Completion Style (for GPT-3.5/4 style models)
def chat_completion(messages):
    """Simulate chat completion with GPT-2"""
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Format messages
    prompt = "\\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
    prompt += "\\nassistant:"
    
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=200,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("assistant:")[-1].strip()

# Example Usage
if __name__ == "__main__":
    # Basic text generation
    use_gpt2()
    
    # Custom parameters
    text = generate_with_params("Once upon a time", max_length=150, temperature=0.8)
    print(text)
    
    # Using pipeline
    use_gpt_pipeline()`
  };

  const tensorflowCode = {
    bert: `import tensorflow as tf
from transformers import TFAutoModel, TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import pipeline

# 1. Using Pre-trained BERT for Classification
def use_bert_classifier():
    """Use pre-trained BERT for sentiment analysis"""
    # Load model and tokenizer
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
    
    # Example text
    text = "I love this product! It's amazing."
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="tf", padding=True, truncation=True)
    
    # Get predictions
    outputs = model(inputs)
    predictions = tf.nn.softmax(outputs.logits, axis=-1)
    
    # Get label
    label_id = tf.argmax(predictions, axis=-1).numpy()[0]
    confidence = predictions[0][label_id].numpy()
    
    labels = ["NEGATIVE", "POSITIVE"]
    print(f"Label: {labels[label_id]}, Confidence: {confidence:.4f}")
    
    return labels[label_id], confidence

# 2. Using Hugging Face Pipeline
def use_pipeline():
    """Use Hugging Face pipeline"""
    # Sentiment Analysis
    classifier = pipeline("sentiment-analysis", framework="tf")
    result = classifier("I love using pre-trained models!")
    print(result)
    
    # Text Generation
    generator = pipeline("text-generation", model="gpt2", framework="tf")
    result = generator("The future of AI is", max_length=50)
    print(result)
    
    # Question Answering
    qa_pipeline = pipeline("question-answering", framework="tf")
    result = qa_pipeline(
        question="What is machine learning?",
        context="Machine learning is a subset of AI..."
    )
    print(result)

# 3. Using BERT for Feature Extraction
def extract_bert_features():
    """Extract features using BERT"""
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TFAutoModel.from_pretrained(model_name)
    
    text = "Hello, how are you?"
    inputs = tokenizer(text, return_tensors="tf", padding=True, truncation=True)
    
    # Get embeddings
    outputs = model(inputs)
    # Use pooled output (CLS token)
    embeddings = outputs.pooler_output
    
    print(f"Embedding shape: {embeddings.shape}")
    return embeddings

# Example Usage
if __name__ == "__main__":
    use_bert_classifier()
    use_pipeline()
    extract_bert_features()`,

    resnet: `import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import ResNet50, ResNet101, ResNet152
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB7
from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
import numpy as np

# 1. Load Pre-trained ResNet Models
def load_resnet_models():
    """Load different ResNet architectures"""
    # ResNet50
    resnet50 = ResNet50(weights='imagenet')
    
    # ResNet101
    resnet101 = ResNet101(weights='imagenet')
    
    # ResNet152
    resnet152 = ResNet152(weights='imagenet')
    
    return resnet50

# 2. Image Classification
def classify_image(model, image_path):
    """Classify image using pre-trained ResNet"""
    # Load and preprocess image
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    # Get predictions
    predictions = model.predict(img_array)
    
    # Decode predictions
    decoded_predictions = decode_predictions(predictions, top=5)[0]
    
    for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
        print(f"{i+1}. {label}: {score:.4f}")
    
    return decoded_predictions

# 3. Feature Extraction
def extract_features(model, image_path):
    """Extract features from image"""
    # Remove classification head
    feature_model = keras.Model(
        inputs=model.input,
        outputs=model.layers[-2].output
    )
    
    # Preprocess image
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    # Extract features
    features = feature_model.predict(img_array)
    
    print(f"Feature shape: {features.shape}")
    return features

# 4. Using EfficientNet
def use_efficientnet():
    """Use EfficientNet for image classification"""
    # EfficientNetB0 (smallest)
    model = EfficientNetB0(weights='imagenet')
    
    # EfficientNetB7 (largest, most accurate)
    # model = EfficientNetB7(weights='imagenet')
    
    return model

# 5. Batch Processing
def classify_batch(model, image_paths):
    """Classify multiple images"""
    images = []
    for path in image_paths:
        img = image.load_img(path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        images.append(img_array)
    
    # Stack and preprocess
    batch = np.stack(images)
    batch = preprocess_input(batch)
    
    # Get predictions
    predictions = model.predict(batch)
    decoded = decode_predictions(predictions, top=1)
    
    return decoded

# Example Usage
if __name__ == "__main__":
    # Load model
    model = load_resnet_models()
    
    # Classify image
    results = classify_image(model, "image.jpg")
    
    # Extract features
    features = extract_features(model, "image.jpg")
    
    # Use EfficientNet
    efficient_model = use_efficientnet()`,

    gpt: `import tensorflow as tf
from transformers import TFAutoModelForCausalLM, AutoTokenizer
from transformers import pipeline

# 1. Using GPT-2 with TensorFlow
def use_gpt2():
    """Use pre-trained GPT-2"""
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = TFAutoModelForCausalLM.from_pretrained("gpt2")
    
    # Set pad token
    tokenizer.pad_token = tokenizer.eos_token
    
    # Input prompt
    prompt = "The future of artificial intelligence"
    
    # Tokenize
    inputs = tokenizer.encode(prompt, return_tensors="tf")
    
    # Generate
    outputs = model.generate(
        inputs,
        max_length=150,
        num_return_sequences=3,
        temperature=0.8,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode
    for i, output in enumerate(outputs):
        text = tokenizer.decode(output, skip_special_tokens=True)
        print(f"Generated {i+1}:\\n{text}\\n")

# 2. Using Pipeline
def use_gpt_pipeline():
    """Use GPT via pipeline"""
    generator = pipeline(
        "text-generation",
        model="gpt2",
        framework="tf"
    )
    
    result = generator(
        "The impact of AI on society",
        max_length=100,
        num_return_sequences=1,
        temperature=0.7
    )
    
    print(result[0]["generated_text"])

# Example Usage
if __name__ == "__main__":
    use_gpt2()
    use_gpt_pipeline()`
  };

  const getCode = () => {
    if (selectedFramework === 'pytorch') {
      return pytorchCode[selectedModel] || pytorchCode.bert;
    } else {
      return tensorflowCode[selectedModel] || tensorflowCode.bert;
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex gap-4 mb-4">
        <button
          onClick={() => setSelectedFramework('pytorch')}
          className={`px-4 py-2 rounded-lg font-semibold transition-colors ${
            selectedFramework === 'pytorch'
              ? 'bg-orange-500 text-white'
              : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
          }`}
        >
          PyTorch
        </button>
        <button
          onClick={() => setSelectedFramework('tensorflow')}
          className={`px-4 py-2 rounded-lg font-semibold transition-colors ${
            selectedFramework === 'tensorflow'
              ? 'bg-orange-500 text-white'
              : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
          }`}
        >
          TensorFlow
        </button>
      </div>

      <div className="mb-4">
        <label className="block text-sm font-semibold text-gray-700 mb-2">
          Select Pre-trained Model
        </label>
        <select
          value={selectedModel}
          onChange={(e) => setSelectedModel(e.target.value)}
          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500"
        >
          <option value="bert">BERT (NLP - Classification, QA)</option>
          <option value="resnet">ResNet (Vision - Image Classification)</option>
          <option value="gpt">GPT-2 (NLP - Text Generation)</option>
        </select>
      </div>

      <div className="bg-gray-900 rounded-lg overflow-hidden">
        <SyntaxHighlighter
          language="python"
          style={vscDarkPlus}
          customStyle={{ margin: 0, borderRadius: '0.5rem' }}
          showLineNumbers
        >
          {getCode()}
        </SyntaxHighlighter>
      </div>

      <div className="bg-blue-50 rounded-lg p-4 border-2 border-blue-200">
        <h3 className="font-semibold text-blue-900 mb-2">Key Concepts:</h3>
        <ul className="list-disc list-inside space-y-1 text-blue-800 text-sm">
          <li><strong>BERT:</strong> Bidirectional encoder for NLP tasks (classification, QA, NER)</li>
          <li><strong>ResNet:</strong> Deep residual networks for image classification</li>
          <li><strong>GPT-2:</strong> Generative pre-trained transformer for text generation</li>
          <li><strong>Hugging Face:</strong> Platform with thousands of pre-trained models</li>
          <li><strong>Inference:</strong> Use models directly without training</li>
        </ul>
      </div>
    </div>
  );
}

