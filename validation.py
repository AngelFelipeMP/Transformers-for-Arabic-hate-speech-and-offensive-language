if __name__ == "__main__":
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)

    dfx = pd.read_csv(config.DATA_PATH + '/' + config.DATASET_DEV, sep='\t', nrows=config.N_ROWS).fillna("none")
    
    df_results = pd.DataFrame(columns=['task',
                                    'epoch',
                                    'transformer',
                                    'max_len',
                                    'batch_size',
                                    'lr',
                                    'accuracy_train',
                                    'f1-macro_train',
                                    'loss_train'
        ]
    )
    for task in tqdm(config.LABELS, desc='TASKS', position=0):
        
        df_train = dfx.loc[dfx[task]>=0]
        
        for transformer in tqdm(config.TRANSFORMERS, desc='TRANSFOMERS', position=1):
            
            best_epoch, max_len, batch_size, drop_out, lr = best_parameters(task, transformer)
            
            df_predictions = train(df_results,
                                    df_train,
                                    task,
                                    transformer,
                                    config.epochs,
                                    best_epoch,
                                    max_len,
                                    batch_size,
                                    drop_out,
                                    lr
            ) 
            ## save model
            
            df_preds['majority_vote'] = df_preds
            df_preds['higher_sun'] = 
            df_preds_&_labels = pd.merge(dfx, df_predictions, left_index=True)
            
            ## I must live the task C as the last because it depend on task A model and task B model!!!