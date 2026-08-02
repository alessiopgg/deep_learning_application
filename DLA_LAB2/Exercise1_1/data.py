from datasets import (
    DatasetDict,
    get_dataset_split_names,
    load_dataset,
)


DATASET_NAME = "cornell-movie-review-data/rotten_tomatoes"


def get_available_split_names(
        dataset_name: str = DATASET_NAME,
) -> list[str]:
    """
    Return the split names declared by the dataset repository.

    This lets us inspect which splits are available before or
    independently from loading the complete DatasetDict.
    """
    return get_dataset_split_names(dataset_name)


def load_rotten_tomatoes(
        dataset_name: str = DATASET_NAME,
) -> DatasetDict:
    """
    Load the complete Cornell Rotten Tomatoes dataset.

    The expected result is a DatasetDict containing the official
    dataset splits.
    """
    dataset = load_dataset(dataset_name)

    if not isinstance(dataset, DatasetDict):
        raise TypeError(
            "Expected load_dataset() to return a DatasetDict, "
            f"but received {type(dataset).__name__}."
        )

    return dataset