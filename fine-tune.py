import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import scanpy as sc
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import json
from transformer import TransformerEncoder



adata = sc.read_h5ad('./data/sc_human_breast_revised.h5ad')

gene_expression = adata.X.A
gene_names = adata.var_names

gene_token_mapping = {gene: i for i, gene in enumerate(gene_names)}
all_tokens = np.array([gene_token_mapping[gene] for gene in gene_names])

max_expression_value = int(np.max(gene_expression))
expression_mapping = {each: i for i, each in enumerate(range(-1, max_expression_value + 1))}

all_expressions = np.array([expression_mapping.values()])



labels = adata.obs["cell_type"]

label_map = {celltype: i for i, celltype in enumerate(labels.unique())}


X_train, X_val = train_test_split(gene_expression, test_size=0.9, random_state=1)


Y_train, Y_val = train_test_split(labels, test_size=0.9, random_state=1)



class SingleCellClassifier(nn.Module):
    def __init__(self, token_size, expression_size, hidden_size, num_classes):
        super(SingleCellClassifier, self).__init__()

        self.hidden_size = hidden_size
        self.token_embedding = nn.Embedding(token_size, hidden_size)
        self.expression_embedding = nn.Embedding(expression_size, hidden_size)

        self.transformer_encoder = TransformerEncoder(hidden_size = hidden_size, nhead=1, ff_size = 4* hidden_size, num_layers = 1)

        self.fc = nn.Linear(hidden_size, 1)

        self.classifier = nn.Linear(1, num_classes)

    def forward(self, token_input, expression_input, mask= None):

        embedded_token = self.token_embedding(token_input)
        
        embedded_expression = self.expression_embedding(expression_input)

        combined_input = embedded_token + embedded_expression

        transformer_output = self.transformer_encoder(combined_input, mask)

        sentence = self.fc(transformer_output)

        pooled_output = sentence.mean(dim=1)
    
        logits = self.classifier(pooled_output)
        return logits
    



class CellTypeDataset(Dataset):
    def __init__(self, expressions, labels):
        self.expressions = expressions
        self.labels = labels
        self.all_gene_indices = np.arange(len(gene_token_mapping))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        expression = self.expressions[index]
        input_tokens = np.array([gene_token_mapping[gene] for gene in gene_names])

        selected_indices= np.random.choice(self.all_gene_indices,500, replace=False)


        expression = expression[selected_indices]
        input_tokens = input_tokens[selected_indices]

        label = label_map[self.labels[index]]



        return {
            'expression': torch.LongTensor(expression),
            'input_tokens': torch.LongTensor(input_tokens),
            'label': torch.tensor(label, dtype=torch.long)
        }
    



num_classes = 19 

model = SingleCellClassifier(
    token_size=len(gene_token_mapping),
    expression_size = len(expression_mapping),
    hidden_size=8,
    num_classes=num_classes
)

# model.load_state_dict(torch.load('./model/pretrain_model_state_dict.pth'))


train_dataset = CellTypeDataset(X_train, Y_train)

train_dataloader = DataLoader(train_dataset, batch_size=6, shuffle=False)


optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()



num_epochs = 1
for epoch in tqdm(range(num_epochs), desc="Epochs"):
    
    model.train()
    total_epoch_loss = 0 

    for batch in train_dataloader:
        input_expression, input_tokens, label = (
        batch['expression'],
        batch['input_tokens'],
        batch['label'],
    )
    
        optimizer.zero_grad()

        
        predict_cell_type = model(input_tokens, input_expression, None)


        # print(predict_cell_type.shape)
 
        # print(label.shape)

      
        loss = criterion(predict_cell_type, label)
        # print(loss)
        loss.backward()
        optimizer.step()

        total_epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_epoch_loss / len(train_dataloader)}")

torch.save(model.state_dict(), './model/finetune_model_state_dict.pth')