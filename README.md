# Glass Identification using Machine Learning

This project aims to classify different types of glass based on their chemical attributes using various machine learning models. It compares the performance of three different classifiers to determine the most effective one for this dataset.


## 📖 About the Dataset

The project uses the "Glass Identification" dataset, which contains 214 instances. The goal is to classify glass into 6 types based on 9 chemical attributes (features).

**Features:**
1.  `RI`: Refractive Index
2.  `Na`: Sodium
3.  `Mg`: Magnesium
4.  `Al`: Aluminum
5.  `Si`: Silicon
6.  `K`: Potassium
7.  `Ca`: Calcium
8.  `Ba`: Barium
9.  `Fe`: Iron

**Target Variable:**
*   `Type`: Type of glass (1 through 7, with no instances of type 4)

## 🤖 Models Implemented

Three different classification algorithms are trained and evaluated:
1.  **K-Nearest Neighbors (KNN)**: A simple, instance-based learning algorithm.
2.  **Gaussian Naive Bayes**: A probabilistic classifier based on Bayes' theorem with an assumption of normality for features.
3.  **Random Forest**: An ensemble learning method that operates by constructing multiple decision trees.

## 🚀 Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

You need Python 3.x installed on your system. You will also need the following libraries:
*   pandas
*   scikit-learn

### Installation

1.  Clone the repository:
    ```sh
    git clone <your-repository-url>
    cd <repository-directory>
    ```
2.  Install the required packages:
    ```sh
    pip install pandas scikit-learn
    ```

## 💻 Usage

To run the script, make sure the `glass.csv` file is in the same directory as `main.py`. Then, execute the following command in your terminal:

```sh
python main.py
```

The script will load the data, train the three models, evaluate their accuracy, and print a summary of the results.