import os, sys, datetime, signal, traceback
import multiprocessing as mp


class DeathBed(Exception):
    pass

class MyKeyboardInterupt(Exception):
    pass

class WhoKilledMe():

    def __init__(self):
        signal.signal(signal.SIGINT, self.interupt_me_not)
        signal.signal(signal.SIGTERM, self.death_on_my_terms)

    def death_on_my_terms(self, sig, frame):
        # Handler for the signal
        print_out("I HAVE BEEN KILLED. WEEP FOR ME.")
        raise DeathBed('Received signal ' + str(sig) +
                  ' on line ' + str(frame.f_lineno) +
                  ' in ' + frame.f_code.co_filename)

    def interupt_me_not(self, sig, frame):
        # Handler for the signal
        print_out("Codus-interuptus. Well, that is annoying.")
        raise MyKeyboardInterupt('Received signal ' + str(sig) +
                  ' on line ' + str(frame.f_lineno) +
                  ' in ' + frame.f_code.co_filename)


def pick_directory(dev_data):
    # Sub-directory to save the data to
    ori = 'unknown'

    if type(dev_data) == str:

        ori = dev_data

    elif type(dev_data) == dict:

        if 'orientation_x_given' in dev_data.keys():

            ori = dev_data['orientation_x_given']

        else:

            ori = dev_data['orientation']

    dir_ext = os.path.realpath('..')

    if ori == 'out_file':

        return dir_ext

    elif ori == 'zz':

        return os.path.join(dir_ext, 'saved_files', 'zz')

    elif ori == 'ac':

        return os.path.join(dir_ext, 'saved_files', 'ac')

    elif ori == 'unknown':

        return os.path.join(dir_ext, 'saved_files')

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


def params_to_txt(file_name, param_dict, extra_str = None, write_type = 'w'):
    """
    Prints all parameters for the current run to a text file and gives the file
    the same name as the corresponding data file

    """
    with open(file_name + '.txt', write_type) as f:

        f.write('\n' + file_name)

        f.write('\n\n\tAll required dictionary keys with their corresponding' +
            ' values...\n')

        max_len = max(len(key) for key in param_dict.keys())

        for key, val in param_dict.items():

            f.write('\n\t' + key.ljust(max_len + 1) + '\t\t' + str(val))

        if extra_str is not None:

            f.write('\n\n\tWith extra data:\n\n\t' + extra_str)


def make_file_name(dir_str, data_str, ext):
    """
    Creates a file name by adding integer values to a base name until a unique
    name is found

    """

    file_name = os.path.join(dir_str, data_str)

    i = 1
    while os.path.exists(file_name + '_' + '%s'.rjust(4, '0') % i + ext):
        i += 1

    file_name += '_' + '%s'.rjust(4, '0') % i

    print_out(str(data_str) + ' data saving to :\n\n\t' + str(file_name) +
        '.extension')

    return file_name


def create_out_file(file_name = 'out.txt'):

    n = datetime.datetime.now()

    f_name = os.path.join(pick_directory('out_file') , file_name)

    if os.path.isfile(f_name):

        sys.exit('\n\tOutput file \'' + f_name + '\' already exists')

    else:

        print_out('\tFile started : ' + n.strftime('\t%Y/%m/%d\t%H:%M:%S'),
            write_type = 'w', is_newline = False,
            target_file_name = f_name)


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



def __main__():

    create_out_file('out_test.txt')

    print_out('Am I working as expected?')

    print(make_file_name(pick_directory('zz'), 'testing', '.h5'))


if __name__ == '__main__':
    
    __main__()