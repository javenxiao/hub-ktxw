Config = {
    "data_path": "asset/dataset/作业数据-waimai_10k.csv",
    "test_size": 0.2,
    "bert_path": r"D:\Download\bert-base-chinese",
    "optimizer": "adam",
    "max_length": 100,
    "epoch": 20,
    "batch_size": 16,
    "num_epochs": 10,
    "learning_rate": 3e-5,
    "dropout": 0.1,
    "model_path": "asset/weight",
    "model_name": "tuned_bert.pth"
}

HOST = "localhost"
PORT = 8000