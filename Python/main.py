import utility_functions as uf
import RF_Pipeline as rf
import XGB_Pipeline as xgb
import plot

def main():
    
    config = uf.get_config()
    logger = uf.Logger()

    kwargs = {
        'config': config,
        'logger': logger
    }

    RF = config['model_settings'].getboolean('run_RF')
    XGB = config['model_settings'].getboolean('run_XGB')
    FF = config['model_settings'].getboolean('frequent_features')
    auto_plot = config['model_settings'].getboolean('auto_plot')

    if RF:
        rf_name = input('What would you like to name the rf model? ')

    if XGB:
        xgb_name = input('What would you like to name the xgb model? ')

    if RF:
        rf_model = rf.RandomForest(**kwargs)
        rf_model.run(rf_name)
        if auto_plot:
            plot_model(rf_name)
        if config['model_settings'].getboolean('frequent_features'):
            rf_model.frequent_features(rf_name)
            plot_model(rf_name, frequent_features=True)

    if XGB:
        xgb_model = xgb.eXtremeGradientBoost(**kwargs)
        xgb_model.run(xgb_name)
        if auto_plot:
            plot_model(xgb_name)
        if config['model_settings'].getboolean('frequent_features'):
            xgb_model.frequent_features(xgb_name)
            plot_model(xgb_name, frequent_features=True)

    # If only frequent features is selected
    if FF and not RF and not XGB:
        rf_model = rf.RandomForest(**kwargs)
        xgb_model = xgb.eXtremeGradientBoost(**kwargs)
                        
        names = [input('Run frequent features on which pickle? (or type "all" for all pickles) ')]
        if names[0].lower() == 'all':
            from glob import glob
            names = [f.split('\\')[-1] for f in glob('Python/picklejar/*.pickle')]
            print(f'Found the following pickles: {names}')
        else:
            while input('More models to analyze? (y/n) ') == 'y':
                name = input('Run frequent features on which pickle? ')
                names.append(name)

        # For each name provided, determine if it's RF or XGB and run frequent features and plotting
        for name in names:
            name = name.replace('.pickle','')
            logger.info(f'Processing {name}...')
            if 'rf' in name.lower():
                rf_model.frequent_features(name)
                plot_model(name, frequent_features=True)
            elif 'xgb' in name.lower():
                xgb_model.frequent_features(name)
                plot_model(name, frequent_features=True)


def plot_model(name, frequent_features=False):
    if frequent_features:
        name = f'{name} Frequent Features'
        path = f'frequent pickles/{name}'
        analyzer = plot.model_analyzer(path, name)
    else:
        analyzer = plot.model_analyzer(name, name)
    analyzer.plot_all()

main()
