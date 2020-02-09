import os
import argparse


def main(args):
    domain_folder = {
        'academic_advising': 'acad',
        'crossing_traffic': 'crossing_traffic',
        'game_of_life': 'game',
        'navigation': 'nav',
        'skill_teaching': 'skill',
        'sysadmin': 'sys',
        'tamarisk': 'tamarisk',
        'traffic': 'traffic',
        'wildfire': 'wild'
    }

    result = {k: {'0': None, '1': None} for k in domain_folder.keys()}
    result5 = {k: {'0': None, '1': None} for k in domain_folder.keys()}

    for domain, folder in domain_folder.items():
        path = args.log_path
        path = os.path.join(path, folder)
        print(path)
        instances = ['5', '6', '7', '8', '9', '10']
        for setting in os.listdir(path):
            path_setting = os.path.join(path, setting)
            print(path_setting)
            if os.path.isdir(path_setting):
                neighbourhood = setting.split('-')[-3]

                instance_results = {}
                with open(os.path.join(path_setting, 'results_compile.csv'),
                          'r') as f:
                    data = f.readlines()

                for line in data[1:]:
                    line_split = line.strip().split(',')
                    for i, instance in enumerate(instances):
                        if instance not in instance_results.keys():
                            instance_results[instance] = line_split[i + 1]
                        else:
                            if float(line_split[i + 1].split('+-')[0]) > float(
                                    instance_results[instance].split('+-')[0]):
                                instance_results[instance] = line_split[i + 1]

                if '1,2,3,5' in path_setting:
                    result5[domain][neighbourhood] = instance_results
                else:
                    result[domain][neighbourhood] = instance_results
    # print(result)
    with open(os.path.join(args.log_path, 'all_results.csv'), 'w') as f:
        f.write('RESULT,FOR,3,,FOR,5\n')
        f.write('Domain,NR_0,NR_1,,NR_0,NR_1\n')
        for domain, res in result.items():
            for instance in instances:
                f.write('{}_{},{},{},,{},{}\n'.format(
                    domain, instance, res['0'][instance].split('+-')[0],
                    res['1'][instance].split('+-')[0],
                    result5[domain]['0'][instance].split('+-')[0],
                    result5[domain]['1'][instance].split('+-')[0]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', required=True)
    args = parser.parse_args()
    main(args)