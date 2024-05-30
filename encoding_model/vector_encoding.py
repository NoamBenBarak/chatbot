import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from config import MODEL_NAME

print("Loading Model...")
model = SentenceTransformer(MODEL_NAME, device="cpu")

print("Loading JSON...")
df = pd.read_json("startups_demo.json", lines=True)

print("Enconding Vectors...")
vectors = model.encode(
    [row.alt + ". " + row.description for row in df.itertuples()],
    show_progress_bar=True,
)

print("Saving Vectors...")
np.save("startup_vectors.npy", vectors, allow_pickle=False)