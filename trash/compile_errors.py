import time
import glob
from collections import defaultdict
import os

def main():

    root = '/data/cajun_results/cajun-complete/' # on doppler

    error_folder = f'{root}/errors/'
    output_folder = f'{root}/errors_summary/'
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    start = time.time()

    errors = compile_errors(error_folder)

    save_error_lists(errors, output_folder)

    count_errors(errors, output_folder)

    print(f'\n{(time.time() - start)} seconds')


def compile_errors(error_folder):

    error_files = glob.glob(f'{error_folder}/*.txt')

    errors = defaultdict(set)
    
    for e, error_file in enumerate(error_files):
        with open(error_file, 'r') as infile:
            text = infile.read()

        text = text.split('\n')
        inds = [i for i in range(len(text)) if text[i].startswith('Radar File')]
        
        for ind in inds:
            scan = text[ind].split('/')[-1]
            error = text[ind+1][12:]  # 12: gets rid of 'Exception - '

            errors[error].add(scan)

        if e % (len(error_files)//3) == 0: print(f'{e}/{len(error_files)} error files finished')

    errors = {key: sorted(list(val)) for key, val in errors.items()}

    return errors


def save_error_lists(errors, output_folder):

    for e, (error, err_list) in enumerate(errors.items()):

        with open(f'{output_folder}/error-{e}_counts-{len(err_list)}.txt', 'w') as outfile:
            outfile.write(error+'\n')
            outfile.write('\n'.join(err_list))


def count_errors(errors, output_folder):

    error_counts = [(error, len(scan_list)) for error, scan_list in errors.items()]
    error_counts = sorted(error_counts, key=lambda x:x[1], reverse=True)
    
    print('\nCOUNT\tERROR')
    for error, count in error_counts:
        print(f'{count} \t{error}')

    with open(f'{output_folder}/error_counts.txt', 'w') as outfile:
        outfile.write('COUNT\tERROR\n')
        
        for error, count in error_counts:
            outfile.write(f'{count} \t{error}\n')


if __name__ == '__main__':
    main()
