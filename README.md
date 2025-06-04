
## YOLO
### Hyperparameter tuning Notebook
`yolo11s-seg-hyperparm-tuning.ipynb`
- YOLO 모델(세그멘테이션)의 하이퍼파라미터 튜닝 노트북
</br>

### Training Notebook
`yolo_train.ipynb`
- YOLO 모델 학습용 노트북

</br></br>


## LLM
### Dataset
`llm_dataset`
- train, eval, test 로 나눠진 LLM 파인튜닝 dataset
</br></br>

### Pre-trained Model
HyperCLOVAX-SEED-Vision-Instruct-3B</br>
https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B
</br></br>

### Fine-tuned Model
`hyperclovax-3b-fine-tuned`
- HyperCLOVAX-SEED-Vision-Instruct-3B 파인 튜닝된 모델(어댑터)
</br></br>

### Fine-tuning Notebook
`hyperclovax-3b-fine-tuning.ipynb`
- HyperCLOVAX-SEED-Vision-Instruct-3B 모델의 파인튜닝 노트북  
</br>

### Result
|  | Pre-trained | Fine-tuned |
|------|-------|-------|
|Accuracy               | 0.5981   | 0.8521   |
|Macro Precision        | 0.5961   | 0.8586   |
|Macro Recall           | 0.5640   | 0.8416   |
|Macro F1-score         | 0.5711   | 0.8491   |
|Avg time per text(sec) | 1.5629   | 1.8610   |
