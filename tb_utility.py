import datetime


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


def create_out_file(file_name = 'out.txt'):

    n = datetime.datetime.now()

    print_out('\tFile started : ' + n.strftime('\t%Y/%m/%d\t%H:%M:%S'),
        write_type = 'w', is_newline = False, target_file_name = file_name)


def __main__():
    
    create_out_file('out_test.txt')

    print_out('Am I working as expected?')


if __name__ == '__main__':
    
    __main__()