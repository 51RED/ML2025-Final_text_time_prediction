
import os
import joblib
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from omegaconf import OmegaConf
from huggingface_hub import login
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding

from utils import parse_arguments, get_device


def process_data(df):
    # Remove outliers
    print("\n" + "="*70)
    print("DATA PREPROCESSING")
    print("="*70)
    
    # Remove extreme values (>= 800 minutes)
    original_len = len(df)
    df = df[df['incident_clearance_duration_minutes'] < 800].copy()
    df = df[df['incident_clearance_duration_minutes'] > 0].copy()

    print(f"Original records: {original_len}")
    print(f"After filtering: {len(df)}")
    print(f"Removed: {original_len - len(df)}")
    # Rename column
    df = df.rename(columns={'duration_minutes': 'incident_clearance_duration_minutes'})

    # Apply log transformation
    df['log_duration'] = np.log1p(df['incident_clearance_duration_minutes'])

    print(f"\nDuration statistics (original):")
    print(df['incident_clearance_duration_minutes'].describe())

    print(f"\nDuration statistics (log-transformed):")
    print(df['log_duration'].describe())
    return df


class IncidentDatasetForLLM_regression(Dataset):
    def __init__(self, texts, labels_minutes, tokenizer, max_length=512):
        self.texts = texts
        self.labels_minutes = labels_minutes
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

        label = torch.tensor([self.labels_minutes[idx]], dtype=torch.bfloat16)

        input_dict = {k: v.squeeze(0) for k, v in encoding.items()}
        input_dict['labels'] = label

        return input_dict
# ============================================================================
# DATASET FOR FINE-TUNING
# ============================================================================

class IncidentDatasetForLLM_generative(Dataset):
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


class Regression_Trainer():
    def __init__(self):
        self.sample_ratio = 1.0
        self.device = get_device()
    def fit(self, config, token, X_train, y_train):
        if torch.cuda.is_available():
            # 這是清空 GPU 顯存快取的標準指令
            torch.cuda.empty_cache()
            print("✓ cuda 顯存快取已清空。")
        elif torch.mps.is_available():
            torch.mps.empty_cache()
            print("mps 顯存快取已清空。")
        else:
            print("unknown device.")

        print("\n" + "="*70)
        print("LOADING LLM MODEL")
        print("="*70)
        print(f"Model: {config.MODEL_NAME}")
        print("\nLoading smaller model (0.5B) to avoid memory issues...")
        
        # Load tokenizer
        llm_tokenizer = AutoTokenizer.from_pretrained(
            config.MODEL_NAME,
            trust_remote_code=True,
            token=token,
        )

        # Set pad token
        if llm_tokenizer.pad_token is None:
            llm_tokenizer.pad_token = llm_tokenizer.eos_token
            print(f"✓ Set pad_token = eos_token, pad_token = {llm_tokenizer.pad_token}")
        llm_tokenizer.padding_side = "left"

        print("\n" + "="*80)
        print("EXPERIMENT 3: LORA FINE-TUNING")
        print("="*80)

        # sample_ratio = 1.0
        num_samples = int(len(X_train) * self.sample_ratio)
        sample_indices = np.random.choice(len(X_train), num_samples, replace=False)

        X_train_sample = X_train[sample_indices]
        y_train_sample = y_train[sample_indices]

        print(f"Using {num_samples} samples for faster training")

        # Create dataset
        train_dataset = IncidentDatasetForLLM_regression(
            texts=X_train_sample,
            labels_minutes=y_train_sample,
            tokenizer=llm_tokenizer,
            max_length=config.MAX_LENGTH
        )
      
        # Dataloader with smaller batch size
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
        )
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Training batches: {len(train_loader)}")

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

        # Optimizer
        optimizer = torch.optim.AdamW(
            finetuned_llm.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        finetuned_llm.train()

        print("\n" + "="*70)
        print("STARTING FINE-TUNING")
        print("="*70)
        total_epoch = config.EPOCHS

        for epoch in range(total_epoch):  
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epoch}")

            for step, batch in enumerate(progress_bar):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = finetuned_llm(**batch)
                loss = outputs.loss
                
                optimizer.zero_grad()

                loss.backward()

                torch.nn.utils.clip_grad_norm_(finetuned_llm.parameters(), max_norm=1.0)

                optimizer.step()

                total_loss += loss.item()
                progress_bar.set_postfix({'MSE': loss.item()})

                if (step + 1) % 50 == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        print("cuda 顯存快取已清空。")
                    elif torch.mps.is_available():
                        torch.mps.empty_cache()
                        print("mps 顯存快取已清空。")
                    else:
                        print("unknown device.")
            avg_loss = total_loss / len(train_loader)
            print(f"\nEpoch {epoch+1} - Average Loss: {avg_loss:.4f}")
                
        print("\n✓ Fine-tuning complete")

        model = finetuned_llm.merge_and_unload()
        print("✓ LoRA weights merged")
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), Path("checkpoints", "llama_3.2_3B_ICT_regression.pth"))


class Generative_Trainer():
    def __init__(self):
        self.sample_ratio = 1.0
        self.device = get_device()
    def fit(self, config, token, X_train, y_train):
        if torch.cuda.is_available():
            # 這是清空 GPU 顯存快取的標準指令
            torch.cuda.empty_cache()
            print("✓ cuda 顯存快取已清空。")
        elif torch.mps.is_available():
            torch.mps.empty_cache()
            print("mps 顯存快取已清空。")
        else:
            print("unknown device.")

        print("\n" + "="*70)
        print("LOADING LLM MODEL")
        print("="*70)
        print(f"Model: {config.MODEL_NAME}")
        print("\nLoading smaller model (0.5B) to avoid memory issues...")

        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            config.MODEL_NAME,
            trust_remote_code=True,
            token=token
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'  # 對於生成任務

        print("\n" + "="*80)
        print("EXPERIMENT 3: LORA FINE-TUNING")
        print("="*80)

        # sample_ratio = 1.0
        num_samples = int(len(X_train) * self.sample_ratio)
        sample_indices = np.random.choice(len(X_train), num_samples, replace=False)

        X_train_sample = X_train[sample_indices]
        y_train_sample = y_train[sample_indices]

        print(f"Using {num_samples} samples for faster training")

        # Create dataset
        train_dataset = IncidentDatasetForLLM_generative(
            texts=X_train_sample,
            labels_minutes=y_train_sample,
            tokenizer=tokenizer,
            max_length=config.MAX_LENGTH
        )
      
        # Dataloader with smaller batch size
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
        )
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Training batches: {len(train_loader)}")

        torch.cuda.empty_cache()

        print("\n✓ Memory cleared")

        finetuned_llm = AutoModelForCausalLM.from_pretrained(
            config.MODEL_NAME,
            torch_dtype=torch.float16 if config.USE_FP16 else torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        finetuned_llm = finetuned_llm.to(self.device)
        print("✓ Model loaded on GPU (no CPU offloading)")
        
        # using Lora
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

        # Optimizer
        optimizer = torch.optim.AdamW(
            finetuned_llm.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        finetuned_llm.train()

        print("\n" + "="*70)
        print("STARTING FINE-TUNING")
        print("="*70)

        for epoch in range(config.EPOCHS):  
            total_loss = 0
            optimizer.zero_grad()
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS}")

            for step, batch in enumerate(progress_bar):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                # Forward pass
                outputs = finetuned_llm(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                loss = loss / config.GRADIENT_ACCUMULATION_STEPS

                # Backward pass
                loss.backward()
                if (step + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(finetuned_llm.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                total_loss += loss.item() * config.GRADIENT_ACCUMULATION_STEPS
                progress_bar.set_postfix({'loss': loss.item() * config.GRADIENT_ACCUMULATION_STEPS})
                
                if (step + 1) % 100 == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        print("cuda 顯存快取已清空。")
                    elif torch.mps.is_available():
                        torch.mps.empty_cache()
                        print("mps 顯存快取已清空。")
                    else:
                        print("unknown device.")
            avg_loss = total_loss / len(train_loader)
            print(f"\nEpoch {epoch+1} - Average Loss: {avg_loss:.4f}")
                
        print("\n✓ Fine-tuning complete")

        model = finetuned_llm.merge_and_unload()
        print("✓ LoRA weights merged")
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), Path("checkpoints", "llama_3.2_3B_ICT_generative.pth"))

if __name__ == "__main__":
    device = get_device()
    args = parse_arguments()
    config = OmegaConf.load(Path("configs", f"{args.method}.yaml"))

    HF_TOKEN = config.MODEL_TOKEN # token
    login(token=HF_TOKEN)
    print("✓ Successfully logged in to Hugging Face")
        
    print("="*70)
    print("LLM EXPERIMENT CONFIGURATION")
    print("="*70)
    print(f"Model: {config.MODEL_NAME}")
    print(f"Device: {device}")
    print(f"FP16: {config.USE_FP16}")
    print(f"LoRA rank: {config.LORA_R}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Gradient accumulation: {config.GRADIENT_ACCUMULATION_STEPS}")
    print(f"Effective batch size: {config.BATCH_SIZE * config.GRADIENT_ACCUMULATION_STEPS}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print("="*70)

    df = pd.read_csv('./data/incident_data_updated_with_incident_clearance_duration_minutes.csv', encoding='utf-8-sig')
    df = process_data(df)
    X = df['raw_report'].values
    y = df['incident_clearance_duration_minutes'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )
    os.makedirs("test_data", exist_ok=True)
    test_data_path = Path("test_data", f"{args.method}_test_data.pkl")
    joblib.dump({'X_test': X_test, 'y_test': y_test}, test_data_path)

    print(f"測試集已成功儲存至 {args.method}_test_data.pkl")
    print(f"訓練集大小: {len(X_train)}")
    print(f"測試集大小: {len(X_test)}")
    print(f"訓練集 incident_clearance_duration_minutes 範圍: {y_train.min():.2f} - {y_train.max():.2f}")
    print(f"測試集 incident_clearance_duration_minutes 範圍: {y_test.min():.2f} - {y_test.max():.2f}")

    if args.method == "regression":
        print("Now we are running regression training.")
        trainer = Regression_Trainer()
    elif args.method == "generative":
        print("Now we are running generative training.")
        trainer = Generative_Trainer()
    else:
        print("unknown task.")

    trainer.fit(config = config, token = HF_TOKEN, X_train = X_train, y_train = y_train)

    
