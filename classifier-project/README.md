# Classifier Project

This project implements a neural network model for classifying images from the Fashion-MNIST dataset using PyTorch. The model is designed to prevent overfitting through the use of dropout layers.

## Project Structure

```
classifier-project
├── src
│   ├── model.py          # Defines the Classifier model with dropout
│   ├── train.py          # Contains the training loop for the model
│   └── utils
│       └── helper.py     # Utility functions for data processing and visualization
├── data                  # Directory for storing datasets
├── tests
│   └── test_model.py     # Unit tests for the model
├── README.md             # Project documentation
└── requirements.txt      # Required Python packages
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd classifier-project
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Download the Fashion-MNIST dataset if not already available. The dataset will be automatically downloaded when running the training script.

## Usage

To train the model, run the following command:
```
python src/train.py
```

After training, you can evaluate the model's performance using the provided test scripts.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.