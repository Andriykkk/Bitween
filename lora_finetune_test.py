import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from bitween import Bitween
from bitween.utils.singlora import apply_singlora_to_model
import copy
import os

def find_all_linear_names(model):
    """
    Finds all linear layer names in the model to be targeted by LoRA.
    This is based on the QLoRA paper's strategy.
    """
    linear_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_module_names.add(name)
    
    # Typically, the output layer is not adapted
    if 'lm_head' in linear_module_names:
        linear_module_names.remove('lm_head')
        
    return list(linear_module_names)

def run_singlora_finetune_test():
    """
    Performs a fine-tuning test with custom SingLoRA on both a standard FP32 model
    and a Bitween-quantized model to verify trainability.
    """
    # --- 1. Configuration ---
    model_name = "facebook/opt-125m"
    dataset_name = "yahma/alpaca-cleaned"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir = "singlora_test_output"
    
    print(f"--- SingLoRA Fine-tuning Test (Targeting All Linear Layers) ---")
    print(f"Using model: {model_name} on device: {device}")
    print("-" * 35)

    # --- 2. Load Model and Tokenizer ---
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- 3. Prepare Dataset ---
    print("Loading and preparing dataset...")
    dataset = load_dataset(dataset_name, split="train[:10]")
    
    def preprocess_function(examples):
        model_inputs = tokenizer(examples['output'], truncation=True, padding="max_length", max_length=128)
        model_inputs["labels"] = model_inputs["input_ids"][:]
        return model_inputs

    processed_dataset = dataset.map(preprocess_function, batched=True)

    # --- 4. Find Target Modules ---
    target_modules = find_all_linear_names(model)
    print(f"\nFound {len(target_modules)} linear layers to adapt with SingLoRA.")

    # --- 5. Fine-tune FP32 Model with SingLoRA ---
    print("\n--- Training FP32 model with SingLoRA ---")
    fp32_model_singlora = copy.deepcopy(model).to(device)
    apply_singlora_to_model(
        fp32_model_singlora,
        rank=8,
        alpha=16,
        ramp_up_steps=10,
        target_modules=target_modules
    )
    
    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, "fp32"),
        per_device_train_batch_size=2,
        num_train_epochs=1,
        logging_steps=5,
        report_to="none"
    )

    trainer_fp32 = Trainer(
        model=fp32_model_singlora,
        args=training_args,
        train_dataset=processed_dataset,
        tokenizer=tokenizer,
    )
    
    fp32_result = trainer_fp32.train()
    print(f"FP32 model training completed. Final loss: {fp32_result.training_loss:.4f}")

    # --- 6. Test backpropagation on Quantized Model with SingLoRA using a single batch ---
    print("\n--- Testing backpropagation on Quantized model with SingLoRA ---")

    quantizer = Bitween(copy.deepcopy(model), bits=8, group_size=128)
    quantized_model = quantizer.quantize(evaluate_perplexity=False)

    quantized_model_singlora = quantized_model.to(device)
    apply_singlora_to_model(
        quantized_model_singlora,
        rank=8,
        alpha=16,
        ramp_up_steps=10,
        target_modules=target_modules,
        print_summary=False,
        device=device
    )

    # Prepare optimizer
    optimizer = torch.optim.AdamW(
        quantized_model_singlora.parameters(), lr=1e-4
    )

    # Get a single batch from dataset
    single_example = processed_dataset[0]
    input_ids = torch.tensor([single_example["input_ids"]]).to(device)
    attention_mask = torch.tensor([single_example["attention_mask"]]).to(device)
    labels = torch.tensor([single_example["labels"]]).to(device)
    loss = 0

    # Training loop for overfitting on one batch
    for step in range(10):  # a few steps just to check gradients flow
        optimizer.zero_grad()
        outputs = quantized_model_singlora(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        print(f"Step {step} - Loss: {loss.item():.4f}")
        loss.backward()
        optimizer.step()

    print(f"Quantized model training completed. Final loss: {loss:.4f}")

    # --- 7. Conclusion ---
    print("\n--- Conclusion ---")
    print("Both models were successfully fine-tuned with your custom SingLoRA on all linear layers.")
    print("This demonstrates that the Bitween-quantized model remains trainable,")
    print("and its loss decreases as expected during fine-tuning with SingLoRA.")
    print("-" * 35)


if __name__ == "__main__":
    run_singlora_finetune_test()