
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
import json
import scanpy as sc
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from finetune import SingleCellClassifier, CellTypeDataset


def loading():
    with open('./model/gene_token_mapping.json', 'r') as f:
        gene_token_mapping = json.load(f)

    with open('./model/expression_mapping.json', 'r') as f:
        expression_mapping = json.load(f)
    
    with open('./model/label_mapping.json', 'r') as f:
        reversed_label_map = json.load(f)


    expression_mapping = {int(k): v for k, v in expression_mapping.items()}
    reversed_label_map = {int(k): v for k, v in reversed_label_map.items()}

    return gene_token_mapping, expression_mapping, reversed_label_map




def predict_method(dataset = './data/demo.h5ad', batch_size = 6, model_path = './model/default_state_dict.pth', out_path = "./predicted_results.csv"):


    gene_token_mapping, expression_mapping, reversed_label_map = loading()



    adata = sc.read_h5ad(dataset)
    X_test = adata.X.A

    num_classes = 19

    model = SingleCellClassifier(
    token_size=len(gene_token_mapping),
    expression_size = len(expression_mapping),
    hidden_size=8,
    num_classes=num_classes
)
    model.load_state_dict(torch.load(model_path))

    test_dataset = CellTypeDataset(X_test)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    model.eval()

    result = []

    for batch in test_dataloader:
        input_expression, input_tokens= (
        batch['expression'],
        batch['input_tokens'],
    )
        
        with torch.no_grad():
            predict_cell_type = model(input_tokens, input_expression, None)
            predictions = torch.argmax(predict_cell_type, dim=1)


            for index in predictions:
                predicted_label = reversed_label_map[index.item()]
                result.append(predicted_label)

    df = pd.DataFrame({'Predicted Cell Type': result})
    df.to_csv(out_path, index=False)




def further_tune(dataset = './data/demo.h5ad', batch_size = 6, learning_rate = 0.001, num_epochs = 1, model_path = './model/further_tune_state_dict.pth'):

    gene_token_mapping, expression_mapping, reversed_label_map = loading()

    adata = sc.read_h5ad(dataset)
    X_train = adata.X.A
    Y_train = adata.obs["cell_type"]

    train_dataset = CellTypeDataset(X_train, Y_train)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)


    num_classes = num_classes = 19

    model = SingleCellClassifier(
    token_size=len(gene_token_mapping),
    expression_size = len(expression_mapping),
    hidden_size=8,
    num_classes=num_classes
)
    model.load_state_dict(torch.load('./model/default_state_dict.pth'))

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    num_epochs =  num_epochs

    
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

            predictions = torch.argmax(predict_cell_type, dim=1)

            for index in predictions:
                predicted_label = reversed_label_map[index.item()]
                print("Predicted Cell Type:", predicted_label)

        
            loss = criterion(predict_cell_type, label)
            loss.backward()
            optimizer.step()

            total_epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_epoch_loss / len(train_dataloader)}")

    torch.save(model.state_dict(), model_path)







