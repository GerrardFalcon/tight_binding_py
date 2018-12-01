import os, sys, datetime, signal, traceback
import multiprocessing as mp

# ---------------------------------- Output ---------------------------------- #

class GenOutFile():

    def __init__(self, directory_IN = "", file_name_IN = 'out.log'):

        file_name = os.path.join(directory_IN, file_name_IN)

        if os.path.isfile(file_name):

            raise Exception('\n\tOutput file \'' + file_name + 
                '\' already exists')

        else:

            self.file_name = file_name


    def __enter__(self):

        self.out_file = open(self.file_name, 'a')

        return self


    def __exit__(self, *exception_args):


        if all(item == None for item in exception_args):

            self.prnt('Closing output file. No errors.')

        else:

            self.prnt('Closing output file. Exceptions : ')

            for arg in exception_args:

                self.prnt(str(arg))
        
        self.out_file.close()


    def pick_directory(self, dev_data):
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


    def prnt(self, str_to_print, is_newline = True):

        if is_newline:

            self.out_file.write('\n\n\t' + str_to_print)

        else:

            self.out_file.write(str_to_print)

        self.out_file.flush()


    def prnt_dict(self, dict_to_print, is_newline = True):

        if is_newline:

            self.out_file.write('\n\n')

        max_len = max(len(key) for key in dict_to_print.keys())

        for key, val in dict_to_print.items():

            self.out_file.write(
                '\n\t' + key.ljust(max_len + 1) + '\t\t' + str(val))

        self.out_file.flush()

# ------------------------------ Error Handling ------------------------------ #

class DeathBed(Exception):
    pass

class MyKeyboardInterupt(Exception):
    pass

class WhoKilledMe():

    def __init__(self, out_file):
        signal.signal(signal.SIGINT, self.interupt_me_not)
        signal.signal(signal.SIGTERM, self.death_on_my_terms)

    def death_on_my_terms(self, sig, frame):
        # Handler for the signal
        raise DeathBed('I HAVE BEEN KILLED. WEEP FOR ME.  Received signal ' + \
            str(sig) + ' on line ' + str(frame.f_lineno) + ' in ' + \
            frame.f_code.co_filename)

    def interupt_me_not(self, sig, frame):
        # Handler for the signal
        raise MyKeyboardInterupt('Codus-interuptus. Received signal ' + \
            str(sig) + ' on line ' + str(frame.f_lineno) + ' in ' + \
            frame.f_code.co_filename)


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
    with open(file_name + '.log', write_type) as f:

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
    while os.path.exists(file_name + '_' + f'{i:03}' + ext):
        i += 1

    file_name += '_' + f'{i:03}'

    return file_name


def __main__():
    """
    create_out_file('out_test.txt')

    print_out('Am I working as expected?')

    print(make_file_name(pick_directory('zz'), 'testing', '.h5'))
    """
    with GenOutFile(pick_directory('out_file'), 'out_test.log') as out_file:

        out_file.prnt('This is working')


if __name__ == '__main__':
    
    __main__()