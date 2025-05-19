import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
from model import parse_description


class LLMExcelApp:
    def __init__(self, root):
        self.root = root
        self.root.title("LLM Excel Parser")
        self.raw_data = []
        self.parsed_data = {}

        tk.Button(root, text="Загрузить Excel",
                  command=self.load_excel).pack(pady=10)
        tk.Button(root, text="Обработать файл",
                  command=self.process_data).pack(pady=10)
        tk.Button(root, text="Сохранить результат",
                  command=self.save_excel).pack(pady=10)

    def load_excel(self):
        file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
        if not file_path:
            return

        try:
            # Загружаем первый столбец, без заголовков
            df = pd.read_excel(file_path, header=None, usecols=[0])
            self.raw_data = df[0].dropna().astype(str).tolist()
            messagebox.showinfo("Успех", f"Загружено строк: {len(self.raw_data)}")
        except Exception as e:
            messagebox.showerror("Ошибка загрузки", str(e))

    def process_data(self):
        if not self.raw_data:
            messagebox.showerror("Ошибка", "Нет загруженных строк")
            return

        self.parsed_data.clear()
        grouped_data = {}

        for desc in self.raw_data:
            parsed = parse_description(desc)
            kv_pairs = {}
            for item in parsed.split("/sprt/"):
                if ":" in item:
                    key, val = item.split(":", 1)
                    kv_pairs[key.strip().lower()] = val.strip()

            category = kv_pairs.pop("категория", "не определено")

            if category not in grouped_data:
                grouped_data[category] = []

            grouped_data[category].append({"Категория": category, **kv_pairs})

        self.parsed_data = {}

        for cat, records in grouped_data.items():
            # Получаем уникальные ключи
            all_keys = set()
            for r in records:
                all_keys.update(r.keys())
            all_keys.discard("Категория")
            sorted_keys = sorted(all_keys)

            # Формируем таблицу
            formatted_rows = []
            for r in records:
                row = {"Категория": r.get("Категория", "не определено")}
                for key in sorted_keys:
                    row[key] = r.get(key, "описания данной характеристики не было")
                formatted_rows.append(row)

            self.parsed_data[cat] = pd.DataFrame(formatted_rows)

        messagebox.showinfo("Готово", "Обработка завершена")

    def save_excel(self):
        if not self.parsed_data:
            messagebox.showerror("Ошибка", "Нет данных для сохранения")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                                 filetypes=[("Excel files", "*.xlsx")])
        if not file_path:
            return

        try:
            with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
                for cat, df in self.parsed_data.items():
                    sheet_name = cat[:31] if cat else "не определено"
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            messagebox.showinfo("Успех", f"Файл сохранен:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Ошибка сохранения", str(e))


# Запуск
if __name__ == "__main__":
    root = tk.Tk()
    app = LLMExcelApp(root)
    root.mainloop()
