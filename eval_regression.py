import os
import json
import torch
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from huggingface_hub import login
from torch.utils.data import Dataset, DataLoader
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification

from utils import parse_arguments, get_device, evaluate_predictions


class RegressionDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding="max_length" 
        )

        return {k: v.squeeze(0) for k, v in encoding.items()}


def batch_predict_regression(texts, labels, model, tokenizer, batch_size=32, verbose=True):
    """
    Batch prediction for a regression model (AutoModelForSequenceClassification with num_labels=1).

    Args:
        texts (list): 事故通報文本列表。
        labels (np.array): 真實標籤 (僅用於顯示，可省略)。
        model: AutoModelForSequenceClassification (已配置 num_labels=1)。
        tokenizer: LLM 分詞器。
        batch_size (int): 批次大小。
        verbose (bool): 顯示進度條。

    Returns:
        np.array: 預測的總處理時間 (分鐘)。
    """

    dataset = RegressionDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    predictions_minutes = []

    model.eval()

    iterator = tqdm(dataloader, desc="Regression Inference") if verbose else dataloader

    with torch.no_grad():
        for batch in iterator:
            inputs = {k: v.to(model.device) for k, v in batch.items()}

            outputs = model(**inputs)

            logits = outputs.logits.to(torch.float32).cpu().numpy().flatten()
            predictions_minutes.extend(logits)

    final_predictions = np.array(predictions_minutes)

    return final_predictions 

class build_model():
    def __init__(self):
        self.device = get_device()
    def tokenizer(self, config, token):
        print("\n" + "="*80)
        print("EVALUATING FINE-TUNED MODEL")
        print("="*80)
        llm_tokenizer = AutoTokenizer.from_pretrained(
                config.MODEL_NAME,
                trust_remote_code=True,
                token=token
            )
        # Set pad token
        if llm_tokenizer.pad_token is None:
            llm_tokenizer.pad_token = llm_tokenizer.eos_token
            print("✓ Set pad_token = eos_token")
        return llm_tokenizer
   
def metric(args, finetuned_test_metrics):
    print("\n" + "="*80)
    print("FINAL COMPARISON: ALL LLM METHODS")
    print("="*80)

    # Create comparison table
    comparison_df = pd.DataFrame({
        'Method': ['Fine-tuned LLM'],
        'MAE (min)': [
            finetuned_test_metrics['mae']
        ],
        'RMSE (min)': [
            finetuned_test_metrics['rmse']
        ],
        'R²': [
            finetuned_test_metrics['r2']
        ],
        'MAPE (%)': [
            finetuned_test_metrics['mape']
        ]
    })

    print("\n")
    print(comparison_df.to_string(index=False))

    csv_path = os.path.join("results", f"comparison_metrics_{args.method}.csv")
    comparison_df.to_csv(csv_path, index=False)
    
    print(f"\n✓ Metrics table saved to {csv_path}")

    # Identify best method
    best_mae_idx = comparison_df['MAE (min)'].argmin()
    best_method = comparison_df.iloc[best_mae_idx]['Method']

    print(f"\n✓ Best performing method: {best_method}")
    print(f"  MAE: {comparison_df.iloc[best_mae_idx]['MAE (min)']:.2f} minutes")

def visualization(args, finetuned_test_pred, finetuned_test_metrics):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    methods = [
        ('Fine-tuned', finetuned_test_pred, finetuned_test_metrics)
    ]

    for idx, (method_name, predictions, metrics) in enumerate(methods):
        axes[idx].scatter(y_test, predictions, alpha=0.5)
        axes[idx].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[idx].set_xlabel('True Duration (minutes)')
        axes[idx].set_ylabel('Predicted Duration (minutes)')
        axes[idx].set_title(f'{method_name} LLM\nMAE: {metrics["mae"]:.2f} min, R²: {metrics["r2"]:.3f}')
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join("results", f"llm_comparison_all_methods_{args.method}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print("\n✓ Visualization saved as 'llm_comparison_all_methods.png'")

def save_result(args, config, finetuned_test_metrics, finetuned_test_pred):

    results_summary = {
        'config': {
            'model': config.MODEL_NAME,
            'lora_r': config.LORA_R,
            'epochs': config.EPOCHS,
            'batch_size': config.BATCH_SIZE,
            'learning_rate': config.LEARNING_RATE
        },
        'finetuned': {
            'metrics': finetuned_test_metrics,
            'predictions': finetuned_test_pred.tolist()
        }
    }

    json_path = os.path.join("results", f"llm_experiment_results{args.method}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Results saved to '{json_path}'")

    print("\n" + "="*80)
    print("LLM EXPERIMENTS COMPLETE!")
    print("="*80)
    print(f"\nAll three LLM approaches have been evaluated:")
    print(f"  3. Fine-tuned: MAE = {finetuned_test_metrics['mae']:.2f} min")
    print(f"\nNext step: Compare these with your BERT baseline results")
    print("="*80)

if __name__ == "__main__":
    # set_random_seed()
    args = parse_arguments()
    config = OmegaConf.load(Path("configs", f"{args.method}.yaml"))

    print("\n" + "="*70)
    print(f"THIS IS evaluation of {config.MODEL_NAME}.")
    print("="*70)

    device = get_device()
    test_data = joblib.load(Path('test_data','regression_test_data.pkl'))
    X_test = test_data['X_test']
    y_test = test_data['y_test']
    HF_TOKEN = config.MODEL_TOKEN
    login(token=HF_TOKEN)
    
    BD_model = build_model()
    llm_tokenizer = BD_model.tokenizer(config = config, token= HF_TOKEN)
    
    finetuned_llm = AutoModelForSequenceClassification.from_pretrained(
                config.MODEL_NAME,
                num_labels=1,                      
                problem_type="regression",          
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
    finetuned_llm = finetuned_llm.to(device)
    model_path = Path("checkpoints", f"llama_3.2_3B_ICT_{args.method}.pth")
    state_dict = torch.load(model_path, map_location=device)

    # 4. 將權重注入模型
    finetuned_llm.load_state_dict(state_dict)
    finetuned_llm.to(device)
    finetuned_llm.eval()

    finetuned_test_pred = batch_predict_regression(
        texts=X_test,
        labels=y_test,
        model=finetuned_llm,
        tokenizer=llm_tokenizer,
        batch_size = 1
    )

    # Evaluate
    finetuned_test_metrics = evaluate_predictions(
        y_test,
        finetuned_test_pred,
        "Fine-tuned LLM Test"
    )

    # Show examples
    print("\n" + "="*70)
    print("EXAMPLE PREDICTIONS (First 5)")
    print("="*70)

    for i in range(min(5, len(X_test))):
        print(f"\n--- Example {i+1} ---")
        print(f"True: {y_test[i]:.0f} min | Predicted: {finetuned_test_pred[i]:.0f} min")
    

    print("\n✓ Fine-tuned model evaluation complete")

    finetuned_results = {
        'predictions': finetuned_test_pred,
        'metrics': finetuned_test_metrics
    }

    # Show results
    os.makedirs("results", exist_ok=True)
    metric(args = args, finetuned_test_metrics = finetuned_test_metrics)
    visualization(
        args = args,
        finetuned_test_pred = finetuned_test_pred,
        finetuned_test_metrics = finetuned_test_metrics
    )
    save_result(
        args = args,
        config = config, 
        finetuned_test_pred = finetuned_test_pred, 
        finetuned_test_metrics = finetuned_test_metrics
    )