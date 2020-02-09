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

    result = {k: {'0': {}, '1': {}} for k in domain_folder.keys()}

    for domain, folder in domain_folder.items():
        path = args.log_path
        path = os.path.join(path, folder)
        print(path)

        for setting in os.listdir(path):
            path_setting = os.path.join(path, setting)
            print(path_setting)
            if os.path.isdir(path_setting):
                neighbourhood = setting.split('-')[-3]
                instance = setting.split('-')[1]

                with open(
                        os.path.join(
                            path_setting,
                            'val_csv/{}{}.csv'.format(domain, instance)),
                        'r') as f:
                    data = f.readlines()[1:]
                    data = [float(a.split(',')[-1]) for a in data]
                max_val = max(data)
                min_val = min(data)
                last_val = data[-1]

                result[domain][neighbourhood][instance] = (min_val, max_val,
                                                           last_val)
    # print(result)
    with open(os.path.join(args.log_path, 'baseline_results.csv'), 'w') as f:

        f.write('Domain,min_value,max_value,last_value\n')
        for domain, res in result.items():
            for instance in ['5', '6', '7', '8', '9', '10']:
                f.write('{}_{},{},{},{}\n'.format(domain, instance,
                                                  res['1'][instance][0],
                                                  res['1'][instance][1],
                                                  res['1'][instance][2]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', required=True)
    args = parser.parse_args()
    main(args)