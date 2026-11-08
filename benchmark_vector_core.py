import time
import numpy as np
import vector_core

class NumPyBaselineIndex:
    """Pure Python/NumPy baseline for exact k-NN comparison."""
    def __init__(self, dimension: int):
        self.dim = dimension
        self.vectors = []
        self.ids = []

    def add_vectors(self, vectors: np.ndarray, ids: np.ndarray):
        self.vectors.extend(vectors)
        self.ids.extend(ids)

    def search(self, query: np.ndarray, k: int):
        v_matrix = np.array(self.vectors)
        # Compute squared Euclidean distance using vectorized NumPy operations
        distances = np.linalg.norm(v_matrix - query, axis=1)
        top_k_indices = np.argsort(distances)[:k]
        return np.array(self.ids)[top_k_indices], distances[top_k_indices]


def run_benchmark():
    print("==================================================")
    print("   VECTOR INDEX CORE: PERFORMANCE BENCHMARK")
    print("==================================================")

    dimension = 128
    num_database_vectors = 10000
    num_queries = 500
    k = 5

    # Generate synthetic dataset
    print(f"Generating {num_database_vectors} database vectors ({dimension}-dim)...")
    db_vectors = np.random.rand(num_database_vectors, dimension).astype(np.float32)
    db_ids = np.arange(num_database_vectors, dtype=np.int64)

    queries = np.random.rand(num_queries, dimension).astype(np.float32)

    # 1. Benchmark C++ VectorIndexCore
    print("\n--- Benchmarking C++20 Core (pybind11) ---")
    cpp_index = vector_core.VectorIndexCore(dimension=dimension)
    
    start_time = time.time()
    cpp_index.add_vectors(db_vectors, db_ids)
    cpp_add_time = time.time() - start_time
    print(f"C++ Indexing Time: {cpp_add_time:.4f} seconds")

    start_time = time.time()
    for q in queries:
        cpp_index.search(q, k)
    cpp_search_time = time.time() - start_time
    cpp_qps = num_queries / cpp_search_time
    cpp_latency_ms = (cpp_search_time / num_queries) * 1000.0

    print(f"C++ Search Time ({num_queries} queries): {cpp_search_time:.4f} seconds")
    print(f"C++ Throughput (QPS): {cpp_qps:.2f} queries/sec")
    print(f"C++ Avg Latency: {cpp_latency_ms:.3f} ms/query")

    # 2. Benchmark Pure NumPy Baseline
    print("\n--- Benchmarking Pure Python / NumPy Baseline ---")
    py_index = NumPyBaselineIndex(dimension=dimension)
    
    start_time = time.time()
    py_index.add_vectors(db_vectors, db_ids)
    py_add_time = time.time() - start_time
    print(f"NumPy Indexing Time: {py_add_time:.4f} seconds")

    start_time = time.time()
    for q in queries:
        py_index.search(q, k)
    py_search_time = time.time() - start_time
    py_qps = num_queries / py_search_time
    py_latency_ms = (py_search_time / num_queries) * 1000.0

    print(f"NumPy Search Time ({num_queries} queries): {py_search_time:.4f} seconds")
    print(f"NumPy Throughput (QPS): {py_qps:.2f} queries/sec")
    print(f"NumPy Avg Latency: {py_latency_ms:.3f} ms/query")

    # 3. Performance Summary
    speedup = py_search_time / cpp_search_time
    print("\n==================================================")
    print(f"RESULT: C++ core is {speedup:.2f}x faster than Python/NumPy baseline!")
    print("==================================================")


if __name__ == "__main__":
    run_benchmark()