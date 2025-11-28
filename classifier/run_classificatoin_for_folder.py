import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# -----------------------------
# Custom PostAttentionLayer
# -----------------------------
import tensorflow as tf
from tensorflow.keras.layers import Dense


class PostAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, **kwargs):
        super(PostAttentionLayer, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.fc = Dense(hidden_dim)
        self.context_vector = self.add_weight(
            name="context_vector",
            shape=(hidden_dim,),
            initializer=tf.keras.initializers.RandomNormal(),
            trainable=True,
        )

    def call(self, inputs):
        hidden = self.fc(inputs)
        hidden = tf.nn.tanh(hidden)
        attention_logits = tf.matmul(hidden, tf.reshape(self.context_vector, [-1, 1]))
        attention_weights = tf.nn.softmax(attention_logits, axis=0)
        weighted_inputs = inputs * attention_weights
        return weighted_inputs, attention_weights

    def get_config(self):
        config = super(PostAttentionLayer, self).get_config()
        config.update({"hidden_dim": self.hidden_dim})
        return config


# -----------------------------
# Load model + class names
# -----------------------------
class_names = [
    "beauty",
    "family",
    "fashion",
    "fitness",
    "food",
    "interior",
    "other",
    "pet",
    "travel",
]

model = load_model(
    "influencer_profiler_best.keras",
    custom_objects={"PostAttentionLayer": PostAttentionLayer},
)


# -----------------------------
# Prediction function
# -----------------------------
def predict_influencer_category(feature_vector, model, class_names):
    if len(feature_vector.shape) == 1:
        feature_vector = np.expand_dims(feature_vector, axis=0)

    predictions = model.predict(feature_vector, verbose=0)
    preds = predictions[0]

    top_indices = np.argsort(preds)[-3:][::-1]

    return {
        "predicted_category": class_names[np.argmax(preds)],
        "confidence": float(np.max(preds)),
        "top_predictions": [(class_names[i], float(preds[i])) for i in top_indices],
    }


# -----------------------------
# Run classifier on all files
# -----------------------------
root_folder = "../../implementation/continuous_representation/combined_features_small/"
rows = []

for actual_category in sorted(os.listdir(root_folder)):
    category_path = os.path.join(root_folder, actual_category)
    if not os.path.isdir(category_path):
        continue

    for filename in os.listdir(category_path):
        if not filename.endswith(".npz"):
            continue

        full_path = os.path.join(category_path, filename)

        try:
            with np.load(full_path, allow_pickle=True) as data:
                feature_vector = data["features"].astype(np.float32)
        except Exception as e:
            print(f"ERROR reading {full_path}: {e}")
            continue

        result = predict_influencer_category(feature_vector, model, class_names)
        top = result["top_predictions"]

        rows.append(
            {
                "filename": filename,
                "actual_category": actual_category,
                "predicted_category": result["predicted_category"],
                "confidence": result["confidence"],
                "top1": top[0][0],
                "top1_prob": top[0][1],
                "top2": top[1][0],
                "top2_prob": top[1][1],
                "top3": top[2][0],
                "top3_prob": top[2][1],
            }
        )

        print(f"Processed: {filename}")


# -----------------------------
# Save CSV
# -----------------------------
df = pd.DataFrame(rows)
df.to_csv("predictions.csv", index=False)

print("\n✓ Done! predictions.csv created successfully.")
