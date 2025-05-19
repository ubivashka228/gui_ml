import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from awq import AutoAWQForCausalLM
from peft import PeftModel

# === Конфигурация ===
MODEL_PATH = "./parse_model"
EXAMPLES_STORE_PATH = "./examples_store.json"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
awq_model = AutoAWQForCausalLM.from_quantized(
    MODEL_PATH,
    device="cuda" if torch.cuda.is_available() else "cpu"
)


# === Загрузка примеров из файла ===
def load_examples_store():
    if os.path.exists(EXAMPLES_STORE_PATH):
        with open(EXAMPLES_STORE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# === Сохранение примеров в файл ===
def save_examples_store(store):
    with open(EXAMPLES_STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, indent=2)


# === Подсчет числа характеристик в output ===
def count_fields(output_str):
    return len([p for p in output_str.split("/sprt/") if ":" in p and p.split(":")[1].strip() != ""])


# === Обновление пула примеров для категории ===
def update_example_store(store, category, input_text, output_text):
    if category not in store:
        store[category] = {}

    new_count = count_fields(output_text)

    # Выбираем тип примера
    type_key = "full" if new_count >= 3 else "partial"

    existing = store[category].get(type_key)
    existing_count = count_fields(existing["output"]) if existing else -1

    # Обновляем, если нет примера или новый пример "лучше"
    if existing is None or new_count > existing_count:
        store[category][type_key] = {"input": input_text, "output": output_text}


# === Формирование few-shot подсказки по категории ===
def build_few_shot_prompt(category, store, input_text):
    prompt = ""
    if category in store:
        examples = []
        if "full" in store[category]:
            examples.append(store[category]["full"])
        if "partial" in store[category]:
            examples.append(store[category]["partial"])

        for ex in examples:
            prompt += f"INPUT: {ex['input']}\nOUTPUT: {ex['output']}{tokenizer.eos_token}\n\n"

    prompt += f"INPUT: {input_text}\nOUTPUT:"
    return prompt


# === Генерация с few-shot и автоматическим пополнением ===
def generate_with_few_shot_dynamic(input_text):
    store = load_examples_store()

    # Используем шаблон без примеров для начального прогноза
    prompt = f"INPUT: {input_text}\nOUTPUT:"
    inputs = tokenizer(prompt, return_tensors="pt").to(awq_model.device)

    with torch.no_grad():
        outputs = awq_model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=False)
    raw_output = decoded[len(prompt):].split("INPUT:")[0].strip()

    # Определяем категорию
    category = "не определено"
    for part in raw_output.split("/sprt/"):
        if part.startswith("Категория:"):
            val = part.split(":", 1)[1].strip()
            if val:
                category = val
            break

    # Обновление пула примеров
    update_example_store(store, category, input_text, raw_output)
    save_examples_store(store)

    # Повторная генерация уже с примерами (если появились)
    final_prompt = build_few_shot_prompt(category, store, input_text)
    final_inputs = tokenizer(final_prompt, return_tensors="pt").to(awq_model.device)

    with torch.no_grad():
        outputs = awq_model.generate(
            **final_inputs,
            max_new_tokens=256,
            do_sample=False,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=False)
    final_output = decoded[len(final_prompt):].split("INPUT:")[0].strip()

    # Повторное обновление, если вдруг финальный лучше
    update_example_store(store, category, input_text, final_output)
    save_examples_store(store)

    return final_output
