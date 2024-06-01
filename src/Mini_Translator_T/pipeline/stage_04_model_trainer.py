from Mini_Translator_T.config.configuration import ConfigurationManager
from Mini_Translator_T.components.model_trainer import modelTrainer
from Mini_Translator_T.logging import logger

STAGE_NAME="Model Trainer"
class ModelTrainerPipeline:
    def __init__(self):
        pass

    def main(self):


        try:
            config = ConfigurationManager()
            get_model_config = config.get_train_model_config()
            model = modelTrainer(config=get_model_config)
            model.initiate_model_trainer()
        except Exception as e:
            raise e


        
if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainerPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e



