# NeuroXOR

# NeuroXOR

**A Lightweight Neural Network for XOR Classification**

---

## 🚀 Project Overview
NeuroXOR is a simple yet powerful feedforward neural network implemented in C++ to classify the XOR function. It demonstrates the core concepts of neural networks, including forward propagation, activation functions, and model persistence (saving/loading).

---

## 📌 Features
- **Multiple Activation Functions**: Supports Sigmoid, ReLU, Leaky ReLU, and Tanh.
- **Xavier/He Initialization**: Optimized weight initialization for faster convergence.
- **Model Persistence**: Save and load trained models using file handling.
- **Structured Neural Network**: Modular design for easy extension.

---

## 🔧 Installation
To compile and run the project, ensure you have a C++ compiler installed (such as g++ or clang++).

```bash
# Compile the program
g++ -o neuroxor neuroxor.cpp -std=c++11

# Run the executable
./neuroxor
```

---

## ⚙️ How It Works
1. **Data Handling**: Uses XOR gate training data.
2. **Weight Initialization**: Implements He initialization.
3. **Forward Propagation**: Computes hidden and output layers using selected activation functions.
4. **Model Persistence**: Saves and loads model weights and biases for reuse.

---

## 🧠 Activation Functions
| Function | Formula |
|----------|---------|
| **Sigmoid** | \( \sigma(x) = \frac{1}{1 + e^{-x}} \) |
| **ReLU** | \( f(x) = max(0, x) \) |
| **Leaky ReLU** | \( f(x) = x > 0 ? x : 0.01x \) |
| **Tanh** | \( f(x) = tanh(x) \) |

---

## 📂 File Structure
```
NeuroXOR/
│── neuroxor.cpp   # Main source code
│── model.dat      # Saved model weights and biases
│── README.md      # Documentation
```

---

## 📜 Example Output
After training, the model should classify XOR inputs accurately:
```bash
Input: (0,0) -> Output: ~0.0
Input: (0,1) -> Output: ~1.0
Input: (1,0) -> Output: ~1.0
Input: (1,1) -> Output: ~0.0
```

---

## 💡 Future Enhancements
- Implement backpropagation and gradient descent for training.
- Support additional activation functions.
- Expand dataset and improve model generalization.

---

## 🤝 Contributing
Feel free to fork this project and submit pull requests! 🚀

---

## 📜 License
This project is open-source and available under the MIT License.

