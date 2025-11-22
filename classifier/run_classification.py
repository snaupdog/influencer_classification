import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

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


def predict_influencer_category(feature_vector, model, class_names):
    if len(feature_vector.shape) == 1:
        feature_vector = np.expand_dims(feature_vector, axis=0)

    predictions = model.predict(feature_vector, verbose=0)

    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_idx]

    confidence = predictions[0][predicted_class_idx]

    top_indices = np.argsort(predictions[0])[-3:][::-1]
    top_predictions = [
        (class_names[idx], float(predictions[0][idx])) for idx in top_indices
    ]

    return {
        "predicted_category": predicted_class,
        "confidence": float(confidence),
        "top_predictions": top_predictions,
    }


sample_file = "../combined_features/combined_feature_vectors/food/asassyspoon-1532724787418428619.npz"
# sample_file = "../../implementation/continuous_representation/combined_features/pet/igpups-1791427037346868330.npz"

with np.load(sample_file, allow_pickle=True) as data:
    feature_vector = data["features"].astype(np.float32)

result = predict_influencer_category(feature_vector, model, class_names)

print(f"Predicted category: {result['predicted_category']}")
print(f"Confidence: {result['confidence']:.2%}")
print("\nTop 3 predictions:")
for category, prob in result["top_predictions"]:
    print(f"  {category}: {prob:.2%}")
