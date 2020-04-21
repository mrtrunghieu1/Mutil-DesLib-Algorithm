# data_folder = 'C:\\Users\Administrator\\Desktop\\Ro_ver2.7\\RotationAlgorithms-master\\data'

# file_list = ['haberman', 'sonar', 'tae']
file_list =['abalone',
 'adult',
 'appendicitis',
 'artificial1',
 'australian',
 'balance',
 'biodeg',
 'blood',
 'breast-cancer',
 'Breast-Tissue',
 'bupa',
 'chess-krvk',
 'cleveland',
 'Colon',
 'conn-bench-vowel',
 'contraceptive',
 'dermatology',
 'DowJones_dj30-1985-2003',
 'duke_as_uci',
 'electricity-normalized',
 'Embryonal',
 'fertility',
 'glass',
 'GM4',
 'haberman',
 'hayes-roth',
 'heart',
 'hepatitis',
 'hill_valley',
 'iris',
 'isolet',
 'led7digit_new',
 'letter',
 'Leukemia',
 'libras',
 'madelon',
 'magic',
 'mammographic',
 'marketing',
 'monk-2_new',
 'multiple-features',
 'musk1',
 'musk2',
 'newthyroid',
 'page-blocks',
 'penbased_new_fix',
 'phoneme',
 'pima',
 'plant_margin',
 'plant_shape',
 'plant_texture',
 'ring1',
 'satimage',
 'shuttle',
 'Skin_NonSkin',
 'sonar',
 'spambase',
 'svmguide2_as_uci',
 'tae',
 'texture',
 'tic-tac-toe',
 'titanic_new',
 'twonorm1',
 'vehicle',
 'vertebral_3C',
 'waveform_wo_noise',
 'waveform_w_noise',
 'wdbc',
 'wine_red',
 'wine',
 'wine_white',
 'yeast']


# result_folder = 'C:\\Users\\Administrator\\Desktop\\Ro_ver2.7\\RotationAlgorithms-master\\result'

import os
from pathlib import Path

root_path = Path(os.path.dirname(os.path.abspath(__file__)))
def join_root_path(path):
    join_path = os.path.join(root_path, path)
    if not os.path.exists(join_path):
        os.makedirs(join_path)
    return join_path

# MODEL_PATH              = join_root_path('model')
# ENVIRONMENT_CONFIG_FILE = os.path.join(root_path, '.env')

result_folder = os.path.join(str(root_path),'result')

data_folder = os.path.join(str(root_path),'data')