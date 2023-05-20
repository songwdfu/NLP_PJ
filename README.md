# NLP_PJ
NLP Course Project Spring 2023

FullModel: mean of head and body sBERT embeddings respectively, concat and apply MLP (external feature not included)

model_ep30.pth: using fnc only. early stopped at 25 epochs, using fnc_train_mean.pkl and fnc_val_mean.pkl, hidden_structure: [128, 128, 64, 32]， Adam lr = 1e-4, decay = 1e-3, history_ep50.pkl

full_ : using fnc + arc. total_train_mean.pkl, total_val_mean.pkl. saved to total_model_ep .pkl, total_history.pkl
