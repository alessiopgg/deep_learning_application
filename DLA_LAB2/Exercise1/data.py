from datasets import DatasetDict, load_dataset


DATASET_NAME = "cornell-movie-review-data/rotten_tomatoes"


def load_rotten_tomatoes(
    dataset_name: str = DATASET_NAME,
) -> DatasetDict:
    """Load the official Rotten Tomatoes train, validation and test splits."""
    dataset = load_dataset(dataset_name)

    if not isinstance(dataset, DatasetDict):
        raise TypeError("Expected a Hugging Face DatasetDict.")

    return dataset
