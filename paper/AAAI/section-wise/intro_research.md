# Introduction Research - 7 Questions

## Abstract (for reference)
Model provenance depends on external evidence---hashes, metadata, logs, or watermarks---which can fail when checkpoints are adapted or records are incomplete. We ask whether provenance can be inferred from weights alone: if a suspect checkpoint descends from a reference model, does training leave a persistent structural signal? We find that modern residual architectures exhibit such a signal. In a two-layer residual branch with input projection W_in and output projection W_out, training couples paired matrices so that W_out W_in exhibits a diagonal-dominance fingerprint: a signed, identity-like component absent from mismatched or random pairings. For deeper branches, we measure the trace on M=W_K...W_1, formalize it with s(M)=|tr(M)|/||M||_F, derive a separation margin, and compute cross-block score matrices that recover residual-branch correspondences and yield checkpoint-level ancestry evidence.

---

## Q1: Modern model checkpoints are copied, fine-tuned, quantized, merged, renamed, and redistributed ✅

**SENTENCES:**
The open-weight ecosystem has grown explosively: Hugging Face now hosts nearly 3 million model repositories---up from under 100,000 in early 2023---with derivatives proliferating faster than originals. When Meta's LLaMA weights leaked in March 2023, the community produced instruction-tuned variants (Alpaca, Vicuna, Koala) within weeks, each costing under $300 to create; a leaked internal Google memo observed that open-source contributors were "doing things with $100 and 13B parameters that we struggle with at $10M and 540B." This rapid, low-cost adaptation means any widely-distributed checkpoint spawns hundreds of fine-tuned, merged, quantized, and renamed descendants whose lineage quickly becomes untraceable.

**CITATIONS:**
1. [HuggingFace hosts ~2.9M models as of May 2026, with >1M Transformers checkpoints]: HuggingFace Models page (https://huggingface.co/models) and HuggingFace Transformers README (https://github.com/huggingface/transformers)
2. [LLaMA leaked March 3, 2023; derivatives Alpaca, Vicuna, Koala appeared within weeks at $100-$300 training cost]: Wikipedia - LLaMA (https://en.wikipedia.org/wiki/LLaMA); SemiAnalysis "Google: We Have No Moat" leaked memo (https://newsletter.semianalysis.com/p/google-we-have-no-moat-and-neither)
3. ["Doing things with $100 and 13B params that we struggle with at $10M and 540B"]: Leaked Google internal document, published by SemiAnalysis, May 2023

**NOTES:**
- The "nearly 3 million models" figure is from live HuggingFace data as of May 2026 - verify closer to submission
- The Google memo quote is from a leaked document; widely cited but note its provenance
- Wikipedia article on LLaMA provides verifiable dates for the leak (March 3, 2023) and subsequent derivatives

---

## Q2: Provenance currently relies on hashes, metadata, logs, or watermarks ✅

**SENTENCES:**
Current approaches to model provenance rely on externally maintained evidence: model cards and metadata registries document training datasets, base models, and licensing terms as structured annotations accompanying checkpoints; experiment tracking platforms such as MLflow and Weights & Biases record run-level artifacts, hyperparameters, and model lineage graphs that link outputs to their training inputs; and cryptographic hashes provide tamper-evident integrity checks for unmodified files. For cases where theft or unauthorized copying is anticipated, watermarking methods embed ownership signals into network parameters or decision boundaries that can later be extracted to verify provenance.

**CITATIONS:**
1. Model cards and metadata standards: Mitchell et al., "Model Cards for Model Reporting," FAT* 2019; Hugging Face Hub model card specification (https://huggingface.co/docs/hub/model-cards) with YAML metadata fields including base_model, datasets, and version lineage.
2. Experiment tracking and artifact lineage: Weights & Biases Artifacts system for run-based provenance tracking with input/output lineage graphs (https://docs.wandb.ai/guides/artifacts/); MLflow Model Registry for versioning and lineage.
3. Neural network watermarking: Nagai et al., "Digital Watermarking for Deep Neural Networks," International Journal of Multimedia Information Retrieval, 2018 (arXiv:1802.02601) - embeds watermarks into model parameters using regularization during training; Szyller et al., "DAWN: Dynamic Adversarial Watermarking of Neural Networks," ACM Multimedia 2021 - dynamic API-level watermarking for model extraction defense.

**NOTES:**
- The sentences are descriptive and neutral, setting up the landscape without criticism
- Model cards are the de facto standard on Hugging Face
- W&B and MLflow are the two dominant experiment tracking platforms
- Chen et al., "Performance Comparison of Contemporary DNN Watermarking Techniques" (arXiv:1811.03713) could be cited as a survey if preferred

---

## Q3: Hashes break under any weight change; metadata/logs require trust; watermarks require prior insertion ✅

**SENTENCES:**
Cryptographic hashes become invalid after any weight modification---including routine quantization or fine-tuning---while metadata and provenance logs can be falsified, lost, or simply never recorded. Watermarks, though actively researched, require deliberate insertion before deployment and remain vulnerable to removal: a recent systematization found that no surveyed DNN watermarking scheme proved robust against fine-tuning, pruning, or model extraction attacks (Lukas et al. 2022). These limitations leave a critical gap: there is no established method to verify provenance for models that were released without embedded watermarks or whose external records are unavailable.

**CITATIONS:**
1. [Watermarks not robust to removal attacks]: Lukas, N., Jiang, E., Li, X., and Kerschbaum, F. "SoK: How Robust is Image Classification Deep Neural Network Watermarking?" IEEE Symposium on Security and Privacy (S&P), 2022. Key finding: "none of the surveyed watermarking schemes is robust in practice."
2. [Watermarks vulnerable to fine-tuning]: Zhang et al. (arXiv 2312.04469) show that "fine-tuning on normal text causes loss of watermarking capabilities."
3. [Generic watermark removal]: Piet et al. (arXiv 2311.04378) demonstrate a black-box attack that "successfully removes watermarks planted by all three [tested] schemes with only minor quality degradation."

**KEY INSIGHT:**
All current provenance methods require either immutability (hashes), trusted record-keeping (metadata/logs), or proactive insertion (watermarks)---none can verify a model that was deployed without preparation or whose records are incomplete.

---

## Q4: We ask: does training itself leave a recoverable trace in model weights? ✅

**SENTENCES:**
Prior intrinsic fingerprinting methods have asked whether weight statistics---means, variances, or histogram features---can identify models, but these aggregate properties of individual matrices lack verification precision and are easily disrupted. Training dynamics research has established that gradient descent enforces near-orthogonal Jacobians in residual networks, yet this work focused on trainability, not on what structural traces the process leaves behind. We ask a different question: does training itself imprint a recoverable signature---not in aggregate statistics, but in the *relational* structure between paired weight matrices?

**CITATIONS:**
1. [Prior intrinsic fingerprint work]: Zheng et al. 2022 (Fingerprinting DNNs via Universal Adversarial Perturbations, CVPR); Zhao et al. 2020 (Shaping Deep Feature Space, CVPR)
2. [Training dynamics/structure work]: Pennington et al. 2017 (Dynamical Isometry, NeurIPS); Saxe et al. 2014 (Exact Solutions in Deep Linear Networks, ICLR); Tarnowski et al. 2019 (Dynamical Isometry in ResNets, AISTATS)

**NOVELTY CLAIM:**
While prior work asked whether trained models have distinguishable statistics (intrinsic fingerprinting) or why residual networks train well (dynamical isometry), we ask whether the optimization process leaves a *structural* imprint in the relational geometry between weight matrices---a question that connects trainability theory to provenance verification.

---

## Q5: Answer: yes, in residual architectures, trained residual-branch products exhibit diagonal-dominant structure ✅

**SENTENCES:**
The answer is yes: in residual architectures, trained residual-branch products exhibit diagonal-dominant structure. This phenomenon emerges because residual blocks must maintain near-identity behavior for stable gradient flow---a requirement formalized as dynamical isometry, where the input-output Jacobian's singular values concentrate near unity throughout training [Pennington et al. 2017, Saxe et al. 2014]. For a residual block computing $x + W_{\text{out}} \phi(W_{\text{in}} x)$, this constraint forces the branch product $W_{\text{out}} W_{\text{in}}$ toward a negative scaled identity: $W_{\text{out}} W_{\text{in}} \approx -\varepsilon I$ [He et al. 2016]. We find this fingerprint is absent at initialization, emerges within the first five epochs of training, and yields a diagonal-dominance score that scales as $\Theta(\sqrt{d})$---separating correctly paired blocks from mismatched pairings by a margin that grows linearly with hidden dimension $d$.

**CITATIONS:**
1. [Dynamical isometry]: Pennington, Schoenholz & Ganguli (2017), "Resurrecting the Sigmoid in Deep Learning through Dynamical Isometry: Theory and Practice", NeurIPS
2. [Deep linear network dynamics]: Saxe, McClelland & Ganguli (2014), "Exact Solutions to the Nonlinear Dynamics of Learning in Deep Linear Neural Networks", ICLR
3. [Residual networks]: He, Zhang, Ren & Sun (2016), "Deep Residual Learning for Image Recognition", CVPR

**FROM ACML PAPER (verified claims):**
- 100% pair accuracy across 9 architecture families (GPT, BERT, LLaMA, Mistral, Qwen, DeepSeek-R1, Gemma, ViT, Whisper)
- Signal absent at init (chance-level ~2% pair accuracy), emerges by epoch 5 with 100% pair accuracy
- Signal scales as O(sqrt(d)): mean s(i,i) rises from 4.18 (GPT-2, d=768) to 7.77 (GPT-2-xl, d=1600)
- Correct pairs at Theta(sqrt(d)), incorrect pairs at Theta(1/sqrt(d)) baseline
- 81-92% of correctly paired products exhibit negative traces, consistent with dynamical isometry
- Non-residual control (PlainNet) achieves only 3% pair accuracy (chance), confirming skip connections are required

---

## Q6: This trace can be used for passive weight-level provenance verification ✅

**SENTENCES:**
This trace enables passive weight-level provenance verification: auditing whether a suspect checkpoint descends from a reference model without requiring watermarks, metadata, or cooperation from the original trainer. Unlike embedded watermarks that must be inserted before or during training, the diagonal-dominance fingerprint emerges from optimization itself and can be read retroactively from any residual model---including checkpoints deployed long before provenance verification was anticipated. The fingerprint maintains 100% pair accuracy across 21 attack configurations (fine-tuning, quantization, pruning, weight noise up to ~20%), degrading only when perturbations also destroy model utility.

**CITATIONS:**
1. [Model auditing/forensics]: Lukas et al. 2022 "SoK: How Robust is Image Classification Deep Neural Network Watermarking?" (IEEE S&P) - establishes that fine-tuning/pruning removes watermarks, motivating passive approaches
2. [Passive verification]: No direct prior work found. Existing approaches (Uchida 2017, Adi 2018, Rouhani 2019) all require explicit watermark insertion. This paper's contribution is novel in the "passive" category.
3. [Weight-level identity vs function-level]: The ACML paper's comparison with membership inference (Shokri 2017) and model extraction (Tramer 2016, Orekondy 2019) clarifies the distinction.

**KEY CAPABILITY (novel claim):**
The diagonal-dominance fingerprint enables retroactive provenance verification for models deployed without watermarks---a capability not offered by any prior method, which all require foresight and modified training.

**METRICS TO HIGHLIGHT:**
- From ACML: 100% pair accuracy across 21 attack configurations; survives fine-tuning (LR 10^-5 to 10^-3, up to 50 epochs), pruning, quantization, weight noise up to ~20%; signal degrades only when model accuracy itself collapses
- From ACML: Distillation erases signal (correct boundary - reads weight-level, not function-level identity)
- From ACML: Independent training does not falsely match (correlation ~0, clean separation from derived models)
- Needs experiment (use X%): Specific AUROC for model-level lineage detection (pair accuracy is reported, but model-level AUROC needs to be computed for the AAAI version)

---

## Q7: Summarize contributions ✅

**CONTRIBUTION BULLETS:**

1. **Discovery**: We identify a training-induced structural trace in residual networks: the product of input and output projections within each block evolves toward a negative scaled identity, $W_{\text{out}} W_{\text{in}} \approx -\varepsilon I$, as gradient descent enforces dynamical isometry along the residual path.
   → Section 3

2. **Formalization**: We derive a closed-form margin showing that the diagonal-dominance score $s(i,j) = |\mathrm{tr}(M)|/\|M\|_F$ separates correctly paired blocks at $\Theta(\sqrt{d})$ from incorrect pairings at the $\Theta(1/\sqrt{d})$ random baseline---a signal-to-noise gap that grows linearly with hidden dimension.
   → Section 3

3. **Method**: We develop a verification protocol that computes cross-block score matrices and applies Hungarian matching to recover residual-branch correspondences, yielding checkpoint-level ancestry evidence from weights alone.
   → Section 4

4. **Evidence**: We validate the fingerprint across nine architecture families (GPT-2 scaling to 1.5B, BERT, LLaMA, Mistral, Qwen, Gemma, ViT, Whisper, ResNet), demonstrate robustness under 21 attack configurations including fine-tuning and quantization, and confirm via ablation that the signal requires both training and residual connections.
   → Section 5, Section 6

**SECTION MAPPING:**
- Section 2: Background
- Section 3: Training-Induced Diagonal Dominance
- Section 4: From Fingerprints to Provenance
- Section 5: Experiments
- Section 6: Adaptive Evasion and Boundaries

