from loguru import logger

from clip_ft.trainer import CLIP_Trainer

from llava.train.train_llm import LLaVA_Trainer
from llava.train.train_utils import parsed_args_train, create_logname_train, get_model, set_seed  

def main():
    model_args, data_args, training_args = parsed_args_train()
    log_name = create_logname_train(data_args, training_args)
    logger.add(log_name)
    logger.info(f"Output saved at file: {log_name}")
    logger.info(f"model_args: {model_args}")
    logger.info(f"data_args: {data_args}")
    logger.info(f"training_args: {training_args}")
    set_seed(training_args.seed)
    
    if 'clip' in training_args.ft_method:
        trainer = CLIP_Trainer(model_args, data_args, training_args, logger)
    else:
        model, tokenizer = get_model(model_args, data_args, training_args)
        trainer = LLaVA_Trainer(model=model, 
                                model_args=model_args, 
                                data_args=data_args, 
                                training_args=training_args, 
                                tokenizer=tokenizer, 
                                lggr=logger, 
                                )
    trainer.train_eval()
           
    
if __name__ == '__main__':
    main()
    