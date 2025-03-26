# 🌟 NeuroXOR: AI-Powered XOR Classifier

🎯 **A Lightweight Neural Network for XOR Classification** 🧠

---

## 🚀 Project Overview
**NeuroXOR** is a compact yet powerful feedforward neural network written in C++ to classify the XOR function. It showcases key neural network concepts such as forward propagation, activation functions, and model persistence (saving/loading). This project is an excellent entry point for learning neural networks in a low-level language!

🛠 **Built With:**
- **C++** – High-performance execution
- **Mathematics** – Applied linear algebra & calculus
- **Machine Learning** – Neural networks & activation functions
- **File Handling** – Save & load trained models

---

## 🌈 Features
✅ **Multiple Activation Functions:** Choose between Sigmoid, ReLU, Leaky ReLU, and Tanh.  
✅ **Optimized Weight Initialization:** Implements **Xavier/He initialization** for better convergence.  
✅ **Model Persistence:** Save & load trained models seamlessly.  
✅ **Structured Neural Network:** Modular, scalable, and easy to extend.  
✅ **Customizable Hyperparameters:** Fine-tune learning rates, epochs, and batch sizes.  

---

## 🔧 Installation & Usage
To compile and run the project, ensure you have a **C++ compiler** (such as g++ or clang++) installed.

```bash
# Compile the program
g++ -o neuroxor neuroxor.cpp -std=c++11

# Run the executable
./neuroxor
```

📌 **Pro Tip:** Experiment with different activation functions and hyperparameters to see how the network behaves!

---

## ⚙️ How It Works
1️⃣ **Data Handling:** Uses XOR gate training data.  
2️⃣ **Weight Initialization:** Implements He initialization for stability.  
3️⃣ **Forward Propagation:** Computes hidden and output layers using the selected activation function.  
4️⃣ **Model Persistence:** Saves and loads model weights & biases to avoid retraining.  

📝 **Mathematical Representation:**
\[
y = f(W_2 \cdot f(W_1 \cdot x + b_1) + b_2)
\]

Where:
- \( W_1, W_2 \) = Weight matrices
- \( b_1, b_2 \) = Bias vectors
- \( f \) = Activation function
- \( x \) = Input vector

---

## 🎨 Activation Functions & Their Formulas
| Function | Formula | Best Use Case |
|----------|---------|--------------|
| **Sigmoid** | \( \sigma(x) = \frac{1}{1 + e^{-x}} \) | Binary classification |
| **ReLU** | \( f(x) = max(0, x) \) | Deep networks |
| **Leaky ReLU** | \( f(x) = x > 0 ? x : 0.01x \) | Avoiding dead neurons |
| **Tanh** | \( f(x) = tanh(x) \) | Zero-centered data |

🔬 **Try different activation functions and observe their impact on training!**

---

## 📂 Project Structure
```
📁 NeuroXOR/
│── 📜 neuroxor.cpp   # Main source code
│── 📄 model.dat      # Saved model weights and biases
│── 📖 README.md      # Documentation
```

---

## 📊 Example Output
After training, the neural network should correctly classify XOR inputs:
```bash
Input: (0,0) -> Output: ~0.0
Input: (0,1) -> Output: ~1.0
Input: (1,0) -> Output: ~1.0
Input: (1,1) -> Output: ~0.0
```

🔍 **Experiment with hyperparameters and observe how accuracy improves!**

---

## 🚀 Future Enhancements
✨ Implement backpropagation & gradient descent.  
✨ Expand dataset & improve generalization.  
✨ Introduce batch training for faster convergence.  
✨ Add visualization support using Python & Matplotlib.  

---

## 🤝 Contributing
Want to improve **NeuroXOR**? Fork the repository, make your changes, and submit a pull request! 🚀

---

## 📜 License
📝 This project is open-source and available under the **MIT License**. Enjoy coding! 🎉

