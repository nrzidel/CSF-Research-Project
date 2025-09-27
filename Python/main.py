import utility_functions as uf
import RF_Pipeline as rf
import XGB_Pipeline as xgb

def main():
    
    config = uf.get_config()
    logger = uf.Logger()

    kwargs = {
        'config': config,
        'logger': logger
    }


    if config['model_settings'].getboolean('run_RF'):
        rf_name = input('What would you like to name the rf model? ')

    if config['model_settings'].getboolean('run_XGB'):
        xgb_name = input('What would you like to name the xgb model? ')

    if config['model_settings'].getboolean('run_RF'):
        rf_model = rf.RandomForest(**kwargs)
        rf_model.run(rf_name)

    if config['model_settings'].getboolean('run_XGB'):
        xgb_model = xgb.eXtremeGradientBoost(**kwargs)
        xgb_model.run(xgb_name)



main()
