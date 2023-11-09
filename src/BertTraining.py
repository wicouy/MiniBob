# Clase: BertTraining.py
import os
import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertForQuestionAnswering, BertTokenizer

from QADataset import QADataset

class BertTraining:
    def __init__(self, script_directory, retraining, num_epochs=3):
        self.script_directory = script_directory
        self.model_path = os.path.join(
            script_directory, "data", "model", "bert_qa_model")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.retraining = retraining
        self.num_epochs = num_epochs
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        if retraining:
            self.model = BertForQuestionAnswering.from_pretrained(
                'bert-base-uncased').to(self.device)
        else:
            try:
                self.model = BertForQuestionAnswering.from_pretrained(
                    self.model_path).to(self.device)
            except Exception as e:
                raise ValueError("No se pudo cargar el modelo desde el path especificado.") from e

    def read_json_file(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return [json.loads(line) for line in file]
        except Exception as e:
            raise ValueError("El archivo JSON no se pudo leer o está mal formateado.") from e

    def read_text_file(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            raise ValueError("El archivo de texto no se pudo leer o no existe.") from e

    def load_and_prepare_data(self, corrections_file, data_file):
        questions = self.read_json_file(corrections_file)
        context = self.read_text_file(data_file)
        return questions, context

    def maybe_train_or_load_model(self):
        if self.retraining:
            corrections_file = os.path.join(
                self.script_directory, "data", "docs", "corrections.jsonl")
            data_file = os.path.join(
                self.script_directory, "data", "docs", "data.txt")
    
            questions, context = self.load_and_prepare_data(
                corrections_file, data_file)
    
            # Verificar si hay preguntas para entrenar
            if not questions:
                print("No hay preguntas para el entrenamiento. Se cargará el modelo existente.")
                self.model = BertForQuestionAnswering.from_pretrained(
                    self.model_path).to(self.device)
                return  # Finalizar la ejecución del método si no hay preguntas
    
            train_dataset = QADataset(
                self.tokenizer, questions, context, max_len=512)
            train_dataloader = DataLoader(
                train_dataset, shuffle=True, batch_size=16)
    
            optimizer = optim.AdamW(self.model.parameters(), lr=5e-5)
    
            self.model.train()
            for epoch in range(self.num_epochs):
                for batch in train_dataloader:
                    optimizer.zero_grad()
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()
    
            self.model.save_pretrained(self.model_path)
        else:
            self.model = BertForQuestionAnswering.from_pretrained(
                self.model_path).to(self.device)

    def predict(self, question, context):
        inputs = self.tokenizer.encode_plus(
            question,
            context,
            add_special_tokens=True,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        answer_start_scores, answer_end_scores = outputs.start_logits, outputs.end_logits
        answer_start = torch.argmax(answer_start_scores)
        answer_end = torch.argmax(answer_end_scores) + 1
        answer_ids = input_ids[0][answer_start:answer_end]

        if len(answer_ids) == 0 or answer_start >= answer_end:
            return "La respuesta no se pudo encontrar en el contexto proporcionado."

        answer_tokens = self.tokenizer.convert_ids_to_tokens(
            answer_ids, skip_special_tokens=True)
        answer = self.tokenizer.convert_tokens_to_string(answer_tokens)

        return answer.strip()
