from ruamel.yaml import YAML
import os
yaml = YAML()
llm_yaml_path = './102344193.yaml'
def get_numbers_from_filenames(directory):
    numbers = []
    for filename in os.listdir(directory):
        if filename.endswith('.scene_instance.json'):
            number = filename.split('.scene_instance')[0]
            numbers.append(number)
    return numbers
# scene_config_directory_path = 'data/scene_datasets/hssd-hab/scenes'
# scene_sample = get_numbers_from_filenames(scene_config_directory_path)
# files_name = ["104862384_172226319","106878960_174887073","108736851_177263586","108736824_177263559","107734254_176000121"]
# files_name = scene_sample
def extract_keys_from_txt(file_path):
    keys = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line or not line.startswith('(') or not line.endswith(')'):
                continue
            try:
                content = line[1:-1].split(',', 1)
                key = content[0].strip().strip("'")
                keys.append(key)
            except Exception as e:
                print(f"Error processing line: {line}, Error: {e}")
    return keys

file_path = 'data_1225_scene.txt'
# keys = extract_keys_from_txt(file_path)
keys = ['102343992', '102815835', '102816036', '102816852', '103997718_171030855', '103997940_171031257', '104348028_171512877', '104862384_172226319', '104862417_172226382', '105515175_173104107', '105515235_173104215', '105515286_173104287', '105515301_173104305', '106366323_174226647', '106878858_174886965', '106879023_174887148', '107733960_175999701', '107734017_175999794', '107734056_175999839', '107734119_175999935', '107734146_175999971', '108294537_176710050', '108294624_176710203', '108294684_176710278', '108294765_176710386', '108294816_176710461', '108294846_176710506', '108294870_176710551', '108294897_176710602', '108294939_176710668', '108736611_177263226', '108736635_177263256', '108736656_177263304', '108736689_177263340', '108736722_177263382', '108736737_177263406', '108736779_177263484', '108736800_177263517', '108736824_177263559', '108736851_177263586', '108736872_177263607', '108736884_177263634']
#failure_yaml
files_name = keys
for file_id in files_name:
    with open(llm_yaml_path,'r') as file:
        data = yaml.load(file)
        for scene_set in data['scene_sets']:
            if scene_set['name'] == 'test':
                scene_set['included_substrings'] = [file_id]
        scene_yaml_path = './allycb_dir'
        if not os.path.exists(scene_yaml_path):
            os.mkdir(scene_yaml_path)
        with open(os.path.join(scene_yaml_path,f'{file_id}.yaml'),'w') as file:
            yaml.dump(data,file)