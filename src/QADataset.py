# Clase: QADataset.py
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

class QADataset(Dataset):
    def __init__(self, tokenizer, questions, context, max_len=512):
        self.tokenizer = tokenizer
        self.questions = questions
        self.context = context  # Este es el texto completo de data.txt
        self.max_len = int(max_len)

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        item = self.questions[idx]
        question = item["input_text"]
        answer_key = f"candidate_{item['choice']}"
        answer = item[answer_key]

        # Codifica la pregunta y el contexto juntos
        encoding = self.tokenizer.encode_plus(
            question,
            self.context,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()
        token_type_ids = encoding['token_type_ids'].flatten()

        # Encuentra los tokens de la respuesta
        answer_tokens = self.tokenizer.encode(answer, add_special_tokens=False)

        # Buscar los índices de inicio y fin de la respuesta en los tokens codificados
        try:
            # Encuentra el inicio y final de la respuesta
            start_positions = input_ids.tolist().index(answer_tokens[0])
            end_positions = start_positions + len(answer_tokens) - 1

            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids,
                'start_positions': torch.tensor(start_positions, dtype=torch.long),
                'end_positions': torch.tensor(end_positions, dtype=torch.long)
            }
        except ValueError:
            # La respuesta no se encuentra en el contexto codificado debido a la truncación
            print(f"La respuesta para la pregunta '{question}' no se encuentra en el contexto.")
            # Puedes manejar este caso como prefieras, por ejemplo, devolviendo la primera posición
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids,
                'start_positions': torch.tensor(0, dtype=torch.long),
                'end_positions': torch.tensor(0, dtype=torch.long)
            }
