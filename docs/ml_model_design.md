# ML Architecture: Design, Inputs, & Outputs

This document breaks down the core Machine Learning architecture housed in src/model.py, detailing the exact tensor flows, the Vision Transformer backbone, and the custom attention pooling mechanism.

## 1. Input Tensor Specifications
The input into the model is heavily structured to provide 3D anatomical context without the immense memory overhead of true 3D Convolutions.

* **Shape:** [Batch_Size, N_Slots, Group_Size, Height, Width]
  * N_Slots = 6 (Standardized anatomical planes: Sagittal, Coronal, etc.)
  * Group_Size = 3 (Three adjacent slices extracted from the center of each slot)
  * Height, Width = 336, 336 (Physical 130mm cropped resolution)

Before passing into the backbone, the tensor is reshaped to [Batch_Size * N_Slots, Group_Size, 336, 336]. The 3 slices (Group_Size) seamlessly act as the 3 RGB channels expected by standard pre-trained Vision models.

## 2. The Backbone: DINOv2
* **Architecture:** HuggingFace dinov2-small.
* **Why DINOv2?** Unlike standard ImageNet supervised models, DINOv2 is trained via self-supervised learning, producing dense, highly generalized spatial embeddings that are incredible for complex medical textures.
* **Surgical Fine-Tuning:** Medical datasets are too small to train Transformers from scratch. To prevent catastrophic forgetting, the architecture aggressively **freezes** the vast majority of the network. Only the **last 6 transformer blocks** and the final LayerNorm are allowed to update their weights (UNFREEZE_LAST = 6).
* **Backbone Output:** The backbone outputs dense contextual embeddings for each of the 6 anatomical slots.

## 3. The SlotHead: Custom Attention Pooling
In a standard classification pipeline, the backbone's 2D feature maps are flattened using Global Average Pooling (GAP). For a complex 3D knee matrix, GAP destroys crucial spatial context.

* **The Design:** The model implements a custom SlotHead attention mechanism.
* **Mechanism:** 
  1. It initializes a learnable "Query" matrix explicitly tailored to the 12 target diseases.
  2. It uses an einsum-based attention layer to dynamically "route" the embeddings from the 6 standardized anatomical slots directly to the relevant disease classifiers. 
  3. *Example:* During training, the attention weights organically learn to heavily query the Sagittal slots when classifying an ACL tear, while querying the Coronal slots for MCL abnormalities.

## 4. Output Tensor Specifications
* **Shape:** [Batch_Size, 12]
* **Targets:** 
  ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', 'Baker's', 'Contusion', 'Fracture']
* **Activation:** The raw logits are passed through a Sigmoid activation function to output final, independent probabilities for each of the 12 anatomical conditions.
