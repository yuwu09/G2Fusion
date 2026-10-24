# Third-party notices

The root MIT license covers original G2Fusion contributions only. It does not replace the terms governing upstream source, models, or datasets. Upstream links below identify projects; the precise revisions used to create this checkout were not recorded.

## DiT

- Project: [facebookresearch/DiT](https://github.com/facebookresearch/DiT), by William Peebles and Saining Xie.
- Copyright: Meta Platforms, Inc. and affiliates.
- License: [Creative Commons Attribution-NonCommercial 4.0 International](LICENSES/DiT-CC-BY-NC-4.0.txt), from the [upstream license](https://github.com/facebookresearch/DiT/blob/main/LICENSE.txt).
- Local files: `config/MP/models/modules/DiT_arch.py` retains upstream code and notices; `config/DiT/models/modules/DiT_arch.py` retains adapted embedding/modulation components.
- Local adaptations include multimodal conditioning, variable-resolution handling, and a dual-stream U-shaped architecture in the standalone DiT branch. The two branches differ.

The retained DiT-derived material has a noncommercial condition. Do not describe the complete release as MIT-only or commercially unrestricted.

## DINOv3 and Transformers

- Projects: [Meta DINOv3](https://github.com/facebookresearch/dinov3) and [Hugging Face Transformers](https://github.com/huggingface/transformers).
- DINOv3 license: [local copy](LICENSES/DINOv3-LICENSE.md), [upstream](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md).
- Transformers license: [Apache-2.0](https://github.com/huggingface/transformers/blob/main/LICENSE).
- Integration: `dino_model/` metadata/fine-tuning helper and `config/*/models/modules/dino_arch.py` wrappers.

DINOv3 weights are not included. Obtain and use the original model under its own terms. The original pretrained model identifier, exact revisions, and helper provenance still require maintainer confirmation. The integration uses local model loading, paired cosine alignment fine-tuning, and patch-feature reshaping.

## OpenAI diffusion implementations

`config/{DiT,MP}/models/modules/diffusion/` identifies adaptations from [GLIDE](https://github.com/openai/glide-text2im), [guided-diffusion](https://github.com/openai/guided-diffusion), and [improved-diffusion](https://github.com/openai/improved-diffusion). Retain the source references in these files.

Copyright (c) 2021 OpenAI. The [MIT license text](LICENSES/OpenAI-diffusion-MIT.txt) is included. Local modifications integrate conditional image-fusion training and sampling.

## Lion optimizer

`config/{Reg,DiT,MP}/models/optimizer.py` contains Google Research's Lion implementation and its existing copyright header (2023 Google Research). See [upstream](https://github.com/google/automl/tree/master/lion) and the included [Apache-2.0 license](LICENSES/Apache-2.0.txt). Existing headers are preserved.

## Additional source references requiring provenance confirmation

Source comments also refer to [pytorch-ssim](https://github.com/Po-Hsun-Su/pytorch-ssim), [cvbase](https://github.com/hellock/cvbase), and [v-diffusion-jax](https://github.com/crowsonkb/v-diffusion-jax). These references appear in SSIM helpers, progress utilities, and positional-embedding utilities. The exact copied portions, original revisions, and required notices are not established by the files provided. Maintainers should confirm that provenance and retain any additional required notices before publication.

External Python packages, datasets, candidate-generation models, and pretrained weights retain their respective licenses. The supplied paper and extracted figure are research documentation; the source-code MIT grant does not establish separate rights for third-party content in them.
