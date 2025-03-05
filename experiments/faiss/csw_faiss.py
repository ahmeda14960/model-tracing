import faiss
import numpy as np
import torch
from typing import Dict, Tuple, List, NamedTuple
import os
import pickle
import yaml
from transformers import AutoModelForCausalLM


class WeightInfo(NamedTuple):
    """
    A named tuple containing metadata about a weight matrix.

    Attributes:
        model_name: Name or identifier of the model
        param_name: Name of the parameter in the model's state dict
        dimensions: Tuple containing the shape of the weight matrix (d1, d2)
    """

    model_name: str
    param_name: str
    dimensions: Tuple[int, int]


class CSWSearch:
    """
    CSWSearch (Cosine Similarity of Weights Search) using FAISS for efficient similarity search.

    This class enables fast indexing and retrieval of similar weight matrices across models,
    organizing weight matrices by their dimensions to ensure comparable searches.
    """

    def __init__(self):
        # Keep track of what each index position corresponds to
        self.metadata: Dict[Tuple[int, int], List[WeightInfo]] = {}
        # Track dimensions and index file locations
        self.index_files: Dict[Tuple[int, int], str] = {}
        # Directory where indices are stored
        self.index_dir: str = "indexes"
        # Currently loaded index
        self.current_index: Tuple[Tuple[int, int], faiss.Index] = None

    def add_weight_matrix(
        self, model_name: str, param_name: str, weight_matrix: np.ndarray
    ) -> None:
        """
        Add a weight matrix to the appropriate index based on its dimensions.

        Args:
            model_name: Name or identifier of the model
            param_name: Name of the parameter in the model's state dict
            weight_matrix: The weight matrix tensor to index

        Returns:
            None
        """
        print(f"Adding {model_name} {param_name}")
        d1, d2 = weight_matrix.shape
        dim_key = (d1, d2)

        # First time seeing this dimension combination
        if dim_key not in self.index_files:
            self.metadata[dim_key] = []
            self.index_files[dim_key] = f"index_{d1}x{d2}.index"

        # Load the appropriate index
        index = self._load_index(dim_key)

        # Flatten matrix in row-major order and normalize
        flat_weights = np.array(weight_matrix.to(dtype=torch.float32).reshape(1, -1).numpy())
        faiss.normalize_L2(flat_weights)  # for cosine similarity

        # Add to appropriate index
        index.add(flat_weights)

        # Store metadata
        self.metadata[dim_key].append(WeightInfo(model_name, param_name, (d1, d2)))

        # Save the updated index
        self._save_index(dim_key, index)

    def find_similar_weights(
        self, model_name: str, weight_matrix: np.ndarray, k: int = 5
    ) -> List[Tuple[WeightInfo, float]]:
        """
        Find similar weight matrices with matching dimensions.

        Searches for weight matrices most similar to the provided one,
        but only among those with the same dimensions.

        Args:
            model_name: Name or identifier of the model (used to exclude self-matches)
            weight_matrix: The weight matrix tensor to search for
            k: Number of similar matrices to return (default: 5)

        Returns:
            List of tuples containing (WeightInfo, similarity_score)

        Raises:
            ValueError: If no weight matrices with matching dimensions are found
        """
        d1, d2 = weight_matrix.shape
        dim_key = (d1, d2)

        if dim_key not in self.index_files:
            raise ValueError(f"No weight matrices found with dimensions {dim_key}")

        # Load the appropriate index
        index = self._load_index(dim_key)

        # Prepare query in same way as stored matrices
        query = np.array(weight_matrix.to(dtype=torch.float32).reshape(1, -1).numpy())
        faiss.normalize_L2(query)

        # Search
        distances, indices = index.search(query, k + 1)  # +1 for self-match

        # Format results (excluding self-match)
        results = []
        for idx, sim in zip(indices[0], distances[0]):
            info = self.metadata[dim_key][idx]
            if info.model_name != model_name:  # Skip self-match
                results.append((info, float(sim)))

        return results[:k]

    def _load_index(self, dim_key: Tuple[int, int]) -> faiss.Index:
        """
        Load or create the FAISS index for a specific dimension.

        Args:
            dim_key: Tuple of dimensions (d1, d2)

        Returns:
            faiss.Index: The loaded or newly created index
        """
        if self.current_index and self.current_index[0] == dim_key:
            return self.current_index[1]

        d1, d2 = dim_key
        index_path = os.path.join(self.index_dir, self.index_files[dim_key])

        if os.path.exists(index_path):
            try:
                index = faiss.read_index(index_path)
            except RuntimeError:
                print(f"Error reading index file {index_path}. Creating a new index.")
                index = faiss.IndexFlatIP(d1 * d2)
        else:
            print(f"Index file {index_path} not found. Creating a new index.")
            index = faiss.IndexFlatIP(d1 * d2)

        self.current_index = (dim_key, index)
        return index

    def _save_index(self, dim_key: Tuple[int, int], index: faiss.Index):
        """
        Save the index for the given dimensions to disk.

        Args:
            dim_key: Tuple of dimensions (d1, d2)
            index: The FAISS index to save

        Returns:
            None
        """
        index_path = os.path.join(self.index_dir, self.index_files[dim_key])
        faiss.write_index(index, index_path)

    def save(self, directory: str):
        """
        Save the entire search system (metadata and indexes) to disk.

        Args:
            directory: Directory where indices and metadata will be stored

        Returns:
            None
        """
        self.index_dir = directory
        os.makedirs(directory, exist_ok=True)

        if self.current_index:
            self._save_index(self.current_index[0], self.current_index[1])

        metadata_path = os.path.join(directory, "metadata.pkl")
        with open(metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)

        index_files_path = os.path.join(directory, "index_files.pkl")
        with open(index_files_path, "wb") as f:
            pickle.dump(self.index_files, f)

    @classmethod
    def load(cls, directory: str):
        """
        Load a previously saved search system from disk.

        Args:
            directory: Directory where indices and metadata are stored

        Returns:
            CSWSearch: The loaded search system
        """
        csw_search = cls()
        csw_search.index_dir = directory

        metadata_path = os.path.join(directory, "metadata.pkl")
        with open(metadata_path, "rb") as f:
            csw_search.metadata = pickle.load(f)

        index_files_path = os.path.join(directory, "index_files.pkl")
        with open(index_files_path, "rb") as f:
            csw_search.index_files = pickle.load(f)

        return csw_search


csw = CSWSearch()


def add_params(model_list):
    """
    Index weight matrices from a list of HuggingFace model IDs.

    Loads each model, extracts its parameters, and adds all 2D weight matrices
    to the CSWSearch index for later similarity search.

    Args:
        model_list: List of HuggingFace model IDs to index

    Returns:
        None: Updates the global csw search index
    """
    for model_id in model_list:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        weights = model.state_dict()
        params = list(weights.keys())
        for param in params:
            # Skip 1D tensors (like bias terms or layer norms)
            if len(weights[param].shape) == 1:
                continue
            csw.add_weight_matrix(model_id, param_name=param, weight_matrix=weights[param])


def get_similar_param(param, k=5):
    """
    Find similar parameters to the given weight matrix across indexed models.

    Args:
        param: Weight matrix tensor to search for
        k: Number of similar matrices to return (default: 5)

    Returns:
        List of tuples containing (WeightInfo, similarity_score)
    """
    return csw.find_similar_weights("--", param, k=k)


def main():
    # Model list to add from yaml
    model_list = yaml.safe_load(open("config/llama7b.yaml", "r"))
    add_params(model_list)
    csw.save("indexes")

    # Weight matrix to search for
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf", torch_dtype=torch.bfloat16
    )
    weights = model.state_dict()
    attn_name = "model.layers.0.self_attn.o_proj.weight"

    print(get_similar_param(weights[attn_name]))

    return


if __name__ == "__main__":
    main()
