import os
import re
import time
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

from utils import parse_arguments, get_device, evaluate_predictions, clear_memory

def extract_number_from_text(response_text):
    """
    從 LLM 回應中提取處理時間（分鐘）
    支援多種格式：
    - 純數字：30、45.5
    - 中文單位：30分鐘、1小時、1小時30分
    - 模糊描述：約30分鐘、大約45分
    - 時間戳：05:59 (解釋為5小時59分)
    """
    if not isinstance(response_text, str):
        return None

    # 清理文本
    text = response_text.strip().lower()

    # 模式1: 尋找「預測時間：數字」或「預測：數字」格式
    pattern1 = r'預測時間\(分鐘數\)[：:]\s*(\d+(?:\.\d+)?)'
    match = re.search(pattern1, text)
    if match:
        return float(match.group(1))

    # 模式2: 尋找「X小時Y分鐘」或「X小時Y分」
    pattern2 = r'(\d+)\s*[小时時]\s*(?:(\d+)\s*分)?'
    match = re.search(pattern2, text)
    if match:
        hours = int(match.group(1))
        minutes = int(match.group(2)) if match.group(2) else 0
        return hours * 60 + minutes

    # 模式3: 尋找「X分鐘」或「X分」
    pattern3 = r'(\d+(?:\.\d+)?)\s*分(?:鐘)?'
    match = re.search(pattern3, text)
    if match:
        return float(match.group(1))

    # 模式4: 尋找時間戳格式 HH:MM (解釋為持續時長)
    pattern4 = r'(\d{1,2}):(\d{2})'
    match = re.search(pattern4, text)
    if match:
        hours = int(match.group(1))
        minutes = int(match.group(2))
        # 只有在合理範圍內才轉換（避免誤判為時間點）
        if hours <= 2:  # 假設處理時間不超過8小時
            return hours * 60 + minutes

    # 模式5: 尋找任何數字（作為最後手段）
    pattern5 = r'(\d+(?:\.\d+)?)'
    matches = re.findall(pattern5, text)
    if matches:
        # 取第一個看起來合理的數字（2-500分鐘範圍）
        for num_str in matches:
            num = float(num_str)
            if 2 <= num <= 500:
                return num
        # 如果都不合理，返回第一個數字
        return float(matches[0])

    return None

print("✓ Extraction function defined")


def save_predictions_to_csv(method_name, X_test, y_test, predictions, generated_texts, filename):
    """Save predictions to CSV for analysis"""
    results_df = pd.DataFrame({
        'raw_report': X_test,
        'true_duration_minutes': y_test,
        'predicted_duration_minutes': predictions,
        'generated_text': generated_texts,
        'absolute_error': np.abs(y_test - predictions)
    })

    results_df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"✓ Saved {method_name} predictions to {filename}")

    return results_df

def predict_duration_zeroshot(text, model, tokenizer):
    """Zero-shot prediction with clearer prompt"""

    prompt = f"""你是專業的事故處理時間預測專家。

事故通報：
{text}

預測的事故處理時間分鐘數:"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,  # 只需要一個數字
            # temperature=0.01,   # 低溫度
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_text = response[len(prompt):].strip()

    predicted_minutes = extract_number_from_text(generated_text)

    if predicted_minutes is None or predicted_minutes <= 0:
        predicted_minutes = np.median(y_train)  # 使用訓練集 median 作為保守預測

    predicted_minutes = np.clip(predicted_minutes, 2, 500)

    return predicted_minutes, generated_text


def batch_predict_zeroshot(texts, labels, model, tokenizer, batch_size=1, verbose=True):
    """
    Batch prediction with zero-shot prompting.

    Args:
        texts (list): List of incident report texts
        labels (np.array): True labels (for display only)
        model: LLM model
        tokenizer: LLM tokenizer
        batch_size (int): Batch size (keep at 1 for generation tasks)
        verbose (bool): Show progress bar

    Returns:
        np.array: Predicted durations in minutes
        list: Generated texts
    """
    predictions_minutes = []
    generated_texts = []

    model.eval()

    iterator = tqdm(texts, desc="Zero-shot Inference") if verbose else texts

    for text in iterator:
        pred_min, gen_text = predict_duration_zeroshot(text, model, tokenizer)
        predictions_minutes.append(pred_min)
        generated_texts.append(gen_text)

    return np.array(predictions_minutes), generated_texts


print("✓ Zero-shot prediction functions defined")

# ============================================================================
# FEW-SHOT PREDICTION
# ============================================================================

def create_fewshot_prompt(text, examples, num_examples=5):
    """
    Create a few-shot prompt with examples.

    Args:
        text (str): Query incident report
        examples (list): List of (text, duration) tuples
        num_examples (int): Number of examples to include

    Returns:
        str: Few-shot prompt
    """
    prompt = "你是專業的事故處理時間預測專家。以下是範例：\n\n"

    for i, (ex_text, ex_duration) in enumerate(examples[:num_examples], 1):
        # 保留完整文本
        prompt += f"事故通報：\n{ex_text}\n預測時間(分鐘數)：{ex_duration:.0f}\n\n"

    # Query 保留完整
    prompt += f"事故通報：\n{text}\n預測的事故處理時間分鐘數:"""

    return prompt


def predict_duration_fewshot(text, examples, model, tokenizer, num_examples=3):
    """
    Predict duration using few-shot prompting.

    Args:
        text (str): Incident report text
        examples (list): List of (text, duration) example tuples
        model: LLM model
        tokenizer: LLM tokenizer
        num_examples (int): Number of examples to use

    Returns:
        float: Predicted duration in minutes
        str: Generated text
    """
    # Create few-shot prompt
    prompt = create_fewshot_prompt(text, examples, num_examples)

    # Tokenize (may need to truncate if prompt is too long)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        # max_length=config.MAX_LENGTH
        max_length=2048
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            # max_new_tokens=config.MAX_NEW_TOKENS,
            max_new_tokens=20,
            # temperature=0.01,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_text = response[len(prompt):].strip()

    # Debugging: Check for empty generation
    if not generated_text:
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        print(f"[DEBUG] Empty generation. Full output: {full_response[-100:]}")

    # Extract number
    predicted_minutes = extract_number_from_text(generated_text)

    if predicted_minutes is None or predicted_minutes <= 0:
        predicted_minutes = np.median(y_train)

    return predicted_minutes, generated_text


def batch_predict_fewshot(texts, labels, examples, model, tokenizer, num_examples=5, verbose=True):
    """
    Batch prediction with few-shot prompting.

    Args:
        texts (list): List of texts to predict
        labels (np.array): True labels
        examples (list): List of (text, duration) example tuples
        model: LLM model
        tokenizer: LLM tokenizer
        num_examples (int): Number of examples per prompt
        verbose (bool): Show progress

    Returns:
        np.array: Predictions
        list: Generated texts
    """
    predictions = []
    generated_texts = []

    model.eval()

    iterator = tqdm(texts, desc="Few-shot Inference") if verbose else texts

    for text in iterator:
        pred, gen_text = predict_duration_fewshot(
            text, examples, model, tokenizer, num_examples
        )
        predictions.append(pred)
        generated_texts.append(gen_text)

    return np.array(predictions), generated_texts

print("✓ Few-shot prediction functions defined")

class IncidentDatasetForLLM(Dataset):
    """
    Dataset for LLM fine-tuning on incident duration prediction.
    """

    def __init__(self, texts, labels_minutes, tokenizer, max_length=512):
        """
        Args:
            texts (list): List of incident report texts
            labels_minutes (np.array): Duration labels in minutes (original scale)
            tokenizer: LLM tokenizer
            max_length (int): Max sequence length
        """
        self.texts = texts
        self.labels = labels_minutes
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Create instruction-following format
        instruction = f"""你是專業的事故處理時間預測專家。

事故通報：
{text}

預測的事故處理時間分鐘數:{label:.0f}"""

        # Tokenize
        encoding = self.tokenizer(
            instruction,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten()  # For language modeling loss
        }


print("✓ LLM Dataset class defined")


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
    
    def LoRA_model(self, config):
        finetuned_llm = AutoModelForSequenceClassification.from_pretrained(
                config.MODEL_NAME,
                num_labels=1,                      
                problem_type="regression",          
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
        finetuned_llm = finetuned_llm.to(self.device)
        print("✓ Model loaded on GPU (no CPU offloading)")
        
        # using Lora
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=config.LORA_R, 
            lora_alpha=config.LORA_ALPHA,
            lora_dropout=config.LORA_DROPOUT,
            target_modules=config.TARGET_MODULES, 
            bias="none"
        )
        finetuned_llm = get_peft_model(finetuned_llm, lora_config)
        print("\n✓ LoRA applied")
        finetuned_llm.print_trainable_parameters()
        return finetuned_llm

def metric(zeroshot_test_metrics, fewshot_test_metrics, finetuned_test_metrics):
    print("\n" + "="*80)
    print("FINAL COMPARISON: ALL LLM METHODS")
    print("="*80)

    # Create comparison table
    comparison_df = pd.DataFrame({
        'Method': ['Zero-Shot LLM', 'Few-Shot LLM', 'Fine-tuned LLM'],
        'MAE (min)': [
            zeroshot_test_metrics['mae'],
            fewshot_test_metrics['mae'],
            finetuned_test_metrics['mae']
        ],
        'RMSE (min)': [
            zeroshot_test_metrics['rmse'],
            fewshot_test_metrics['rmse'],
            finetuned_test_metrics['rmse']
        ],
        'R²': [
            zeroshot_test_metrics['r2'],
            fewshot_test_metrics['r2'],
            finetuned_test_metrics['r2']
        ],
        'MAPE (%)': [
            zeroshot_test_metrics['mape'],
            fewshot_test_metrics['mape'],
            finetuned_test_metrics['mape']
        ]
    })

    print("\n")
    print(comparison_df.to_string(index=False))

    csv_path = os.path.join("results", "comparison_metrics.csv")
    comparison_df.to_csv(csv_path, index=False)
    
    print(f"\n✓ Metrics table saved to {csv_path}")

    # Identify best method
    best_mae_idx = comparison_df['MAE (min)'].argmin()
    best_method = comparison_df.iloc[best_mae_idx]['Method']

    print(f"\n✓ Best performing method: {best_method}")
    print(f"  MAE: {comparison_df.iloc[best_mae_idx]['MAE (min)']:.2f} minutes")

def visualization(args, zero_shot_results, few_shot_results, finetuned_results):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    zeroshot_test_pred = zero_shot_results['predictions']
    zeroshot_test_metrics = zero_shot_results['metrics']
    fewshot_test_pred = few_shot_results['predictions']
    fewshot_test_metrics = few_shot_results['metrics']
    finetuned_test_pred = finetuned_results['predictions']
    finetuned_test_metrics = finetuned_results['metrics']

    methods = [
        ('Zero-Shot', zeroshot_test_pred, zeroshot_test_metrics),
        ('Few-Shot', fewshot_test_pred, fewshot_test_metrics),
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

def save_result(config, zero_shot_results, few_shot_results, finetuned_results):
    zeroshot_test_pred = zero_shot_results['predictions']
    zeroshot_test_metrics = zero_shot_results['metrics']
    fewshot_test_pred = few_shot_results['predictions']
    fewshot_test_metrics = few_shot_results['metrics']
    finetuned_test_pred = finetuned_results['predictions']
    finetuned_test_metrics = finetuned_results['metrics']
    results_summary = {
        'config': {
            'model': config.MODEL_NAME,
            'lora_r': config.LORA_R,
            'epochs': config.EPOCHS,
            'batch_size': config.BATCH_SIZE,
            'learning_rate': config.LEARNING_RATE
        },
        'zeroshot': {
            'metrics': zeroshot_test_metrics,
            'predictions': zeroshot_test_pred.tolist(),
            'inference_time': zero_shot_results['inference_time']
        },
        'fewshot': {
            'metrics': fewshot_test_metrics,
            'predictions': fewshot_test_pred.tolist(),
            'inference_time': few_shot_results['inference_time']
        },
        'finetuned': {
            'metrics': finetuned_test_metrics,
            'predictions': finetuned_test_pred.tolist()
        }
    }

    json_path = os.path.join("results", f"llm_experiment_results_{args.method}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Results saved to '{json_path}'")

    print("\n" + "="*80)
    print("LLM EXPERIMENTS COMPLETE!")
    print("="*80)
    print(f"\nAll three LLM approaches have been evaluated:")
    print(f"  1. Zero-Shot: MAE = {zeroshot_test_metrics['mae']:.2f} min")
    print(f"  2. Few-Shot: MAE = {fewshot_test_metrics['mae']:.2f} min")
    print(f"  3. Fine-tuned: MAE = {finetuned_test_metrics['mae']:.2f} min")
    print(f"\nNext step: Compare these with your BERT baseline results")
    print("="*80)

def zero_shot_prediction(model, tokenizer, X_test, y_test):
        # ============================================================================
    # EXPERIMENT 1: ZERO-SHOT PREDICTION
    # ============================================================================
    
    clear_memory()

    print("\n" + "="*80)
    print("EXPERIMENT 1: ZERO-SHOT LLM PREDICTION")
    print("="*80)
    print(f"Model: {config.MODEL_NAME}")
    print(f"Test set size: {len(X_test)}")
    print("\nThis will take ~10-15 minutes for 449 samples (processing one at a time)")
    print("Progress bar will show estimation...")

    # Record start time
    start_time = time.time()

    # Run zero-shot prediction on test set
    zeroshot_test_pred, zeroshot_generated_texts = batch_predict_zeroshot(
        texts=X_test.tolist() if hasattr(X_test, 'tolist') else list(X_test),
        labels=y_test,
        model=model,
        tokenizer=tokenizer,
        batch_size=1,
        verbose=True
    )

    # Record end time
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"\n✓ Inference completed in {elapsed_time/60:.2f} minutes")
    print(f"  Average time per sample: {elapsed_time/len(X_test):.2f} seconds")

    # Evaluate
    zeroshot_test_metrics = evaluate_predictions(
        y_test,
        zeroshot_test_pred,
        "Zero-Shot LLM Test"
    )

    # Show example predictions
    print("\n" + "="*70)
    print("EXAMPLE PREDICTIONS (First 10)")
    print("="*70)

    for i in range(min(10, len(X_test))):
        print(f"\n--- Example {i+1} ---")
        print(f"Text: {X_test[i][:120]}...")
        print(f"True: {y_test[i]:.0f} min | Predicted: {zeroshot_test_pred[i]:.0f} min | Error: {abs(y_test[i] - zeroshot_test_pred[i]):.0f} min")
        print(f"Generated: '{zeroshot_generated_texts[i]}'")

    print("\n✓ Zero-shot experiment complete")

    # Save results for later comparison
    zeroshot_results = {
        'predictions': zeroshot_test_pred,
        'generated_texts': zeroshot_generated_texts,
        'metrics': zeroshot_test_metrics,
        'inference_time': elapsed_time
    }
    zeroshot_df = save_predictions_to_csv(
        "results",
        X_test, y_test,
        zeroshot_test_pred,
        zeroshot_generated_texts,
        "llm_zeroshot_predictions.csv"
    )
    return zeroshot_results

def few_shot_prediction(model, tokenizer, X_train,y_train,X_test,y_test):
        # ============================================================================
    # EXPERIMENT 2: FEW-SHOT PREDICTION
    # ============================================================================

    clear_memory()

    print("\n" + "="*80)
    print("EXPERIMENT 2: FEW-SHOT LLM PREDICTION")
    print("="*80)
    print(f"Number of examples per prompt: {config.NUM_FEW_SHOT_EXAMPLES}")

    # Create example pool from training set
    # Select diverse examples (different duration ranges)
    train_durations = y_train
    sorted_indices = np.argsort(train_durations)

    # Sample examples from different quantiles
    quantiles = [0.1, 0.3, 0.5, 0.7, 0.9]
    example_indices = [sorted_indices[int(q * len(sorted_indices))] for q in quantiles]

    fewshot_examples = [(X_train[i], y_train[i]) for i in example_indices]

    print(f"\nSelected {len(fewshot_examples)} diverse examples:")
    for i, (text, duration) in enumerate(fewshot_examples, 1):
        print(f"  Example {i}: {duration:.0f} minutes (text length: {len(text)} chars)")

    # Run few-shot prediction
    start_time = time.time()

    fewshot_test_pred, fewshot_generated_texts = batch_predict_fewshot(
        texts=X_test.tolist() if hasattr(X_test, 'tolist') else list(X_test),
        labels=y_test,
        examples=fewshot_examples,
        model=model,
        tokenizer=tokenizer,
        num_examples=config.NUM_FEW_SHOT_EXAMPLES,
        verbose=True
    )

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"\n✓ Inference completed in {elapsed_time/60:.2f} minutes")

    # Evaluate
    fewshot_test_metrics = evaluate_predictions(
        y_test,
        fewshot_test_pred,
        "Few-Shot LLM Test"
    )

    # Show examples
    print("\n" + "="*70)
    print("EXAMPLE PREDICTIONS (First 5)")
    print("="*70)

    for i in range(min(5, len(X_test))):
        print(f"\n--- Example {i+1} ---")
        print(f"True: {y_test[i]:.0f} min | Predicted: {fewshot_test_pred[i]:.0f} min")
        print(f"Generated: '{fewshot_generated_texts[i]}'")

    print("\n✓ Few-shot experiment complete")

    # Save results
    fewshot_results = {
        'predictions': fewshot_test_pred,
        'generated_texts': fewshot_generated_texts,
        'metrics': fewshot_test_metrics,
        'inference_time': elapsed_time
    }
    # After few-shot experiment
    fewshot_df = save_predictions_to_csv(
        "results",
        X_test, y_test,
        fewshot_test_pred,
        fewshot_generated_texts,
        "llm_fewshot_predictions.csv"
    )
    return fewshot_results

def finetune_prediction(finetuned_llm,tokenizer, X_test,y_test):
        # ============================================================================
    # EVALUATE FINE-TUNED MODEL
    # ============================================================================

    clear_memory()

    print("\n" + "="*80)
    print("EVALUATING FINE-TUNED MODEL")
    print("="*80)

    # Use zero-shot style prompting with fine-tuned model
    finetuned_test_pred, finetuned_generated_texts = batch_predict_zeroshot(
        texts=X_test.tolist() if hasattr(X_test, 'tolist') else list(X_test),
        labels=y_test,
        model=finetuned_llm,
        tokenizer=tokenizer,
        verbose=True
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
        print(f"Generated: '{finetuned_generated_texts[i]}'")

    print("\n✓ Fine-tuned model evaluation complete")

    # Save results
    finetuned_results = {
        'predictions': finetuned_test_pred,
        'generated_texts': finetuned_generated_texts,
        'metrics': finetuned_test_metrics
    }
    # After fine-tuning
    finetuned_df = save_predictions_to_csv(
        "Fine-tuned",
        X_test, y_test,
        finetuned_test_pred,
        finetuned_generated_texts,
        "llm_finetuned_predictions.csv"
    )
    return finetuned_results

if __name__ == "__main__":
    # set_random_seed()
    args = parse_arguments()
    config = OmegaConf.load(Path("configs", f"{args.method}.yaml"))

    print("\n" + "="*70)
    print(f"THIS IS evaluation of {config.MODEL_NAME}.")
    print("="*70)

    target = np.load(Path("checkpoints", f"{config.MODEL_NAME}_ICT.pth"))
    target = target.astype(np.float32)
    device = get_device()
    test_data = joblib.load('test_data.pkl')
    X_test = test_data['X_test']
    y_test = test_data['y_test']
    X_train = test_data['X_train']
    y_train = test_data['y_train']
    HF_TOKEN = config.MODEL_TOKEN
    login(token=HF_TOKEN)

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'  # 對於生成任務

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )

    # Show results
    os.makedirs("results", exist_ok=True)

    zero_shot_results = zero_shot_prediction(
        model= model, 
        tokenizer=tokenizer, 
        X_test=X_test,
        y_test=y_test
    )
    few_shot_results = few_shot_prediction(
        model = model, 
        tokenizer = tokenizer, 
        X_train = X_train, 
        y_train = y_train, 
        X_test = X_test, 
        y_test = y_test
    )

    finetuned_llm = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        torch_dtype=torch.float16 if config.USE_FP16 else torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    # 手動移到 GPU
    finetuned_llm = finetuned_llm.to(device)
    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=config.TARGET_MODULES,
        bias="none"
    )

    finetuned_llm = get_peft_model(finetuned_llm, lora_config)
    print("\n✓ LoRA applied")
    finetuned_llm.print_trainable_parameters()

    model_path = Path("checkpoints", f"llama_3.2_3B_ICT_{args.method}.pth")
    state_dict = torch.load(model_path, map_location=device)
    # 4. 將權重注入模型
    finetuned_llm.load_state_dict(state_dict)
    finetuned_llm.to(device)
    finetuned_llm.eval()

    
    finetuned_results = finetune_prediction(
        model = finetuned_llm,
        tokenizer = tokenizer,
        X_test = X_test,
        y_test = y_test
    )

    metric(
        zeroshot_test_metrics = zero_shot_results['metrics'], 
        fewshot_test_metrics = few_shot_results['metrics'], 
        finetuned_test_metrics = finetuned_results['metrics']
    )
    visualization(
        args=args,
        zero_shot_results = zero_shot_results,
        few_shot_results = few_shot_results,
        finetuned_results = finetuned_results,
    )
    save_result(
        config = config,
        zero_shot_results = zero_shot_results,
        few_shot_results = few_shot_results,
        finetuned_results = finetuned_results,
    )