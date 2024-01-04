from src.data_processing.data_loader import load_data
from src.model_training.random_forest_model import train_random_forest

def main():
    # Load data
    data = load_data()

    # Train and evaluate the random forest model
    train_random_forest(data)

if __name__ == "__main__":
    main()
