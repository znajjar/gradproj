import datetime
import json

from dateutil.parser import parse as parse_date

DATA_PATH = 'runs'


# write_data('uni', 'a.png', 3, mean=[1, 2, 3], std=[3, 4, 5])
def write_data(algorithm_name: str, filename: str, iterations_count: int, **data):
    file_path = f'{DATA_PATH}/{algorithm_name}.json'
    with open(file_path, mode='a+', encoding='utf-8') as data_file:
        data_file.seek(0)
        data_json = data_file.read()
    if data_json:
        algorithm_data = json.loads(data_json)
    else:
        algorithm_data = {
            'algorithm': algorithm_name,
            'runs': [],
            'index': {},
        }

    algorithm_data['runs'].append({
        'filename': filename,
        'date': datetime.datetime.utcnow().isoformat(),
        'iterations': iterations_count,
        **{label: serialize_data(data) for label, data in data.items()},
    })

    algorithm_data['index'][filename] = len(algorithm_data['runs']) - 1

    with open(file_path, mode='w', encoding='utf-8') as data_file:
        json.dump(algorithm_data, data_file, indent=3)


def serialize_data(data):
    # return data
    return ','.join(str(float(x)) for x in data)


def read_data(algorithm_name):
    file_path = f'{DATA_PATH}/{algorithm_name}.json'
    with open(file_path, mode='r', encoding='utf-8') as data:
        data = json.load(data)['runs']
        for run_data in data:
            for label, value in run_data.items():
                if label == 'date':
                    run_data[label] = parse_date(value)
                elif label not in ('filename', 'iterations'):
                    run_data[label] = deserialize_data(value)
        return data


def deserialize_data(data):
    # return data
    return [float(x) for x in data.split(',')]

# write_data('uni', 'a.png', mean=[1, 2, 3], std=[4, 5, 6])
