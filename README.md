
# Mitigating Hallucination in Multimodal LLMs with Layer Contrastive Decoding

This repository provides the implementation of **Layer Contrastive Decoding (LayerCD)**, along with evaluation scripts on the **POPE** dataset.

---

## ⚙️ Environment Setup

1. **Clone the Repository**

```bash
git clone git@github.com:maifoundations/LayerCD.git
cd LayerCD
```

2. **Configure Environment**

Set up the environment according to the requirements of the model you want to use with LayerCD (e.g., LLaVA, Cambrian, Molmo). Please refer to the documentation of your chosen model for installation instructions.

3. **Benchmarks**

If you plan to use the [**POPE**](https://github.com/RUCAIBox/POPE.git) benchmark:

- Download the POPE image dataset.
- Update the `IMAGE_BASE` path in [`util/constant.py`](util/constant.py).

4. **Model Weights**

- Update the `MODEL_ZOO` dictionary in [`util/constant.py`](util/constant.py) with the paths to your model checkpoints.

5. **Using Custom Models**

- To apply LayerCD to your own model, check the function `evolve_cd_sampling` in [`util/cd_utils.py`](util/cd_utils.py).
- Modify the image feature extraction logic to match your model’s visual encoder.

------

## 🚀 Running Evaluation

Example: running evaluation on **POPE**:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python eval.py \
  --dataset=POPE \                    # Dataset: POPE or MME
  [--POPE_sampling_type=coco] \       # POPE sampling set (required for POPE)
  [--POPE_type=popular] \             # POPE data type (required for POPE)
  --batch_size=8 \                    # Inference batch size
  --model_type=Molmo \                # Model type: LLaVA, Cambrian, Molmo, or custom
  --seed=$seed                        # Random seed
```

------

## 📊 Computing Results

After evaluation, compute the final results with:

```bash
python util/compute_results.py --dataset=POPE   # Dataset: POPE or MME
```

------

## 📌 Citation

If you find this work useful, please consider citing:

```
TBD
```