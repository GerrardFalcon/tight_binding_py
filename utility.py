import os, sys, datetime
import multiprocessing as mp

def pick_directory(ori):
    # Sub-directory to save the data to

    dir_ext = os.path.realpath('..')

    if ori == 'out_file':

        return dir_ext

    elif ori == 'zz':

        return os.path.join(dir_ext, 'saved_files', 'zz')

    elif ori == 'ac':

        return os.path.join(dir_ext, 'saved_files', 'ac')

    else:

        return os.path.join(dir_ext, 'saved_files', 'other')


def cpu_num(is_main_task, max_cores, **kwargs):

    cpu_no = mp.cpu_count()
    
    if is_main_task:

        if cpu_no < max_cores:

            return cpu_no

        else:

            return max_cores
    
    else:
    
        if cpu_no <= 3:

            return 2
    
        elif cpu_no > 3:

            if cpu_no // 2 < max_cores:

                return cpu_no // 2

            else:

                return max_cores

        else:

            return 1


def time_elapsed_str(time):
    """ Makes a formated string for the time elapsed during the calculation """

    if time > 0 and time < 60:

        return ' %d seconds' % time

    elif time >= 60 and time < 3600:

        return ' %d minutes and %d seconds' % divmod(time, 60)

    elif time >= 3600:

        return ' %d hours and %d minutes' % divmod(time // 60, 60)

    else:

        return ' invalid time entered for \'time_elapsed\'' + time


def make_file_name(dir_str, data_str, param_dict, extra_str = None):
    """ Adds dictionary values to the naming string """

    file_name = os.path.join(dir_str, data_str)

    for key, value in param_dict.items():
        file_name += '_' + key + '_' + str(value)

    if extra_str is not None:
        file_name += '_' + extra_str

    replacements = ['.', '(', ')', ',', '__']

    for rep in replacements:

        file_name = file_name.replace(rep, '_')

    file_name = file_name.replace('-', 'm').replace('+', 'p')

    print_out(str(data_str) + ' data saving to :\n\n\t' + str(file_name))

    return file_name


def create_out_file(file_name = 'out.txt'):

    n = datetime.datetime.now()

    print_out('\tFile started : ' + n.strftime('\t%Y/%m/%d\t%H:%M:%S'),
        write_type = 'w', is_newline = False,
        target_file_name = os.path.join(pick_directory('out_file') , file_name))


def print_out(str_to_print, write_type = 'a', is_newline = True,
    target_file_name = None, file_name = []):
    
    # This is a hack to make it so that I do not need to pass the file name

    if target_file_name is not None:

        file_name.append(str(target_file_name))

    # Save the given string to the file

    with open(file_name[0], write_type) as f:

        if is_newline:

            f.write('\n\n\t' + str_to_print)

        else:

            f.write(str_to_print)


def data_save(data, data_type_str, file_str_ex = None):

    all_params = {**dev.get_req_params(), **pot.get_req_params()}

    file_name = make_file_name(
        pick_directory(all_params['ori']),
        data_type_str,
        {**dev.get_req_params(), **pot.get_req_params()},
        file_str_ex)

    np.savetxt(file_name + '.csv', data, delimiter = ',')



def __main__():
    
    create_out_file('out_test.txt')

    print_out('Am I working as expected?')


if __name__ == '__main__':
    
    __main__()