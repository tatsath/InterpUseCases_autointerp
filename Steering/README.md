# Financial LLM Feature Steering

This directory contains tools for interactive feature steering of the finetuned Llama2-7b-Finance model using Sparse Autoencoder (SAE) features.

## 🎯 Overview

Feature steering allows you to manipulate specific features in the model to influence its behavior on financial text. This is particularly useful for:

- **Risk Assessment**: Adjusting how the model perceives financial risk
- **Sentiment Analysis**: Influencing market sentiment interpretation
- **Investment Decisions**: Steering decision-making biases
- **Financial Analysis**: Enhancing specific analytical capabilities

## 📁 Files

### **Streamlit App**
- **`streamlit_feature_steering_app.py`** - Interactive web application for feature steering
- **`requirements.txt`** - Python dependencies for the Streamlit app
- **`feature_steering_demo.py`** - Original command-line demonstration script

## 🚀 Quick Start

### **1. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **2. Run the Streamlit App**
```bash
streamlit run streamlit_feature_steering_app.py
```

### **3. Access the Web Interface**
Open your browser to `http://localhost:8501`

## 🎛️ Features

### **Interactive Controls**
- **Layer Selection**: Choose from layers 4, 10, 16, 22, 28
- **Feature Selection**: Select from top 10 features per layer based on finetuning impact
- **Steering Strength**: Adjust feature activation from -2.0 to +2.0
- **Real-time Analysis**: See activation changes and model output differences

### **Visualization**
- **Activation Comparison**: Bar charts showing original vs steered activations
- **Model Output**: Side-by-side comparison of original and steered responses
- **Feature Information**: Detailed feature labels and activation improvements

## 🔧 Technical Details

### **Model Configuration**
- **Base Model**: `cxllin/Llama2-7b-Finance`
- **SAE Model**: `llama2_7b_finance_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun`
- **Device**: CUDA (if available) or CPU
- **Max Tokens**: 512 input, 100 output

### **Feature Data**
The app uses the top 10 features per layer from the finetuning impact analysis:
- **Layer 4**: Basic financial processing features
- **Layer 10**: Risk and sentiment features  
- **Layer 16**: Advanced financial operations
- **Layer 22**: Quantitative modeling features
- **Layer 28**: Strategic planning features

### **Steering Mechanism**
- **Activation Modification**: Direct manipulation of SAE feature activations
- **Temperature Adjustment**: Secondary effect through generation temperature
- **Real-time Processing**: Immediate feedback on steering effects

## 📊 Usage Examples

### **Risk Assessment Steering**
1. Select Layer 10, Feature 17 (Risk perception)
2. Set steering strength to +1.0 (increase risk awareness)
3. Input: "The market shows strong bullish trends"
4. Observe: Model becomes more cautious about market risks

### **Sentiment Analysis Steering**
1. Select Layer 10, Feature 173 (Market sentiment)
2. Set steering strength to -0.8 (bearish bias)
3. Input: "Company reports record profits"
4. Observe: Model interprets positive news more cautiously

### **Investment Decision Steering**
1. Select Layer 28, Feature 384 (Decision bias)
2. Set steering strength to +1.5 (aggressive bias)
3. Input: "Consider investing in emerging markets"
4. Observe: Model becomes more optimistic about investment opportunities

## 🔍 Understanding Results

### **Activation Metrics**
- **Original Activation**: Base feature activation level
- **Steered Activation**: Modified activation after steering
- **Activation Change**: Difference between steered and original

### **Model Output Analysis**
- **Original Output**: Model response without steering
- **Steered Output**: Model response with feature steering applied
- **Key Differences**: Look for changes in tone, confidence, and reasoning

## ⚠️ Important Notes

1. **Model Loading**: First run may take time to load the model
2. **Memory Usage**: Ensure sufficient GPU/CPU memory for model inference
3. **Steering Effects**: Results are simulated for demonstration purposes
4. **Feature Selection**: Only top 10 features per layer are available

## 🛠️ Troubleshooting

### **Common Issues**
- **CUDA Out of Memory**: Reduce max_tokens or use CPU
- **Model Loading Errors**: Check model path and dependencies
- **SAE Weight Errors**: Verify SAE model path exists

### **Performance Tips**
- Use shorter input texts for faster processing
- Close other applications to free up memory
- Consider using CPU if GPU memory is limited

## 📈 Future Enhancements

- [ ] Real-time SAE weight modification
- [ ] Batch processing capabilities
- [ ] Advanced visualization options
- [ ] Feature interaction analysis
- [ ] Custom feature selection
- [ ] Export functionality for results

## 🤝 Contributing

To contribute to this project:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is part of the InterpUseCases_autointerp repository.
